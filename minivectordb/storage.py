"""表存储层：列式存储 + WAL + 崩溃恢复 + 事务撤销。

磁盘布局（全部位于 /data 下）：

    /data/catalog.json                 表结构清单
    /data/commit.log                   提交点记录，每次提交追加一行 JSON
    /data/tables/<table>/wal.bin       行元信息日志（row_id / pk / 删除标记）
    /data/tables/<table>/cols/<c>.dat  列数据，槽位下标 = row_id，8 字节定长
    /data/tables/<table>/cols/<c>.str  VARCHAR 字符串堆（追加写）
    /data/tables/<table>/vec.f32       向量块，偏移 = row_id * dim * 4

设计要点：
    * 列为定长槽位，按 row_id 分段读取；不做整列载入，避免大表撑爆内存。
    * 行版本 row_id 单调递增；更新一条记录即分配新 row_id 写新槽位，
      旧槽位自然失效，回滚时按提交点截断文件即可。
    * 向量以定长块存放，同样按连续段读取；扫描与建索引均分块流式处理。
    * commit.log 每次提交只追加一行，覆盖所有表的文件长度；启动时截断到
      最后一个提交点，因此「已确认提交不丢、未提交不部分生效」。
    * 恢复是可重入的：截断与重放都是幂等操作，重复恢复不会改变结果。
"""

import json
import os
import struct

import numpy as np

LENGTH_PREFIX = struct.Struct(">I")
SLOT_WIDTH = 8
# VARCHAR 槽位：('off','<u4'),('len','<u4')
VARCHAR_SLOT = np.dtype([("off", "<u4"), ("len", "<u4")])
BIGINT_SLOT = np.dtype("<i8")

# 一次 pread 的字节上限，避免单次读入过多数据
READ_CHUNK = 1 << 20
# 向量连续段的最大空洞：空洞小于该值时仍合并为一次读
GAP_THRESHOLD = 64
# 向量常驻内存的字节上限；超过就只走磁盘分块读取
VECTOR_CACHE_LIMIT = 64 << 20
# 分块读写的向量条数
CHUNK = 4096


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ColumnStore:
    """单个普通列的定长槽位存储。

    全部使用无缓冲的文件描述符（os.pwrite / os.pread / os.ftruncate），
    避免 Python 缓冲层在「扩展文件 + 定位写入」场景下的位置歧义。
    """

    def __init__(self, directory, name, ctype):
        self.name = name
        self.ctype = ctype
        self.is_text = ctype == "varchar"
        self.data_path = os.path.join(directory, name + ".dat")
        self.str_path = os.path.join(directory, name + ".str")
        self.data_fd = None
        self.str_fd = None
        self.data_len = 0
        self.str_len = 0

    def open(self):
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        self.data_fd = os.open(self.data_path, os.O_RDWR | os.O_CREAT, 0o644)
        self.data_len = os.fstat(self.data_fd).st_size
        if self.is_text:
            self.str_fd = os.open(self.str_path, os.O_RDWR | os.O_CREAT, 0o644)
            self.str_len = os.fstat(self.str_fd).st_size

    def close(self):
        for fd in (self.data_fd, self.str_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.data_fd = None
        self.str_fd = None

    def truncate(self, data_len, str_len):
        self.data_len = data_len
        self.str_len = str_len
        if self.data_fd is not None:
            os.ftruncate(self.data_fd, data_len)
        if self.str_fd is not None:
            os.ftruncate(self.str_fd, str_len)

    def write(self, row_id, value):
        """写入槽位；返回写入后的长度。

        使用 os.pwrite 直接按偏移写入，不经过 Python 缓冲层，
        避免「扩展文件后定位与写入位置不一致」的问题。
        """
        if self.is_text:
            if isinstance(value, bytes):
                payload = value
            else:
                payload = str(value).encode("utf-8")
            offset = self.str_len
            os.pwrite(self.str_fd, payload, offset)
            self.str_len += len(payload)
            slot = np.array([(offset, len(payload))], dtype=VARCHAR_SLOT)
            raw = slot.tobytes()
        else:
            raw = np.array([value], dtype=BIGINT_SLOT).tobytes()
        offset = row_id * SLOT_WIDTH
        os.pwrite(self.data_fd, raw, offset)
        target = offset + SLOT_WIDTH
        if target > self.data_len:
            # 稀疏槽位：补零保证文件长度覆盖到该 row_id
            current = os.fstat(self.data_fd).st_size
            if target > current:
                os.pwrite(self.data_fd, b"\x00" * (target - current), current)
            self.data_len = target
        return self.data_len

    def flush(self):
        for fd in (self.data_fd, self.str_fd):
            if fd is not None:
                os.fsync(fd)

    # ---- 读取 ----

    def _slots_for(self, row_ids):
        """返回 (去重后的 row_id 数组, 对应槽位数组, 每个请求行在数组中的位置)。

        只读取 row_id 落在文件范围内的部分，并按连续段合并 pread，
        既避免整列载入内存，也避免逐行系统调用。
        """
        dtype = VARCHAR_SLOT if self.is_text else BIGINT_SLOT
        empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=dtype),
                 np.zeros(0, dtype=np.int64))
        if self.data_fd is None:
            return empty
        ids = np.unique(np.asarray(row_ids, dtype=np.int64))
        limit = self.data_len // SLOT_WIDTH
        ids = ids[ids < limit]
        if len(ids) == 0:
            return empty
        out = np.zeros(len(ids), dtype=dtype)
        breaks = np.nonzero(np.diff(ids) != 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(ids) - 1]))
        for begin, end in zip(starts, ends):
            low = int(ids[begin])
            high = int(ids[end])
            raw = os.pread(self.data_fd, (high - low + 1) * SLOT_WIDTH, low * SLOT_WIDTH)
            got = np.frombuffer(raw, dtype=dtype, count=high - low + 1)
            out[begin:end + 1] = got
        positions = np.searchsorted(ids, np.asarray(row_ids, dtype=np.int64))
        positions = np.clip(positions, 0, max(len(ids) - 1, 0))
        return ids, out, positions

    def _read_text(self, slots):
        """按合并后的区间读取字符串堆，返回与 slots 等长的 bytes 列表。"""
        count = len(slots)
        if count == 0 or self.str_fd is None:
            return [b""] * count
        offsets = slots["off"].astype(np.int64)
        lengths = slots["len"].astype(np.int64)
        order = np.argsort(offsets, kind="stable")
        merged = []          # [start, end)
        for index in order:
            start = int(offsets[index])
            end = start + int(lengths[index])
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                if end > merged[-1][1]:
                    merged[-1][1] = end
            else:
                merged.append([start, end])
        if not merged:
            return [b""] * count
        starts = np.array([item[0] for item in merged], dtype=np.int64)
        blobs = []
        for start, end in merged:
            blobs.append(os.pread(self.str_fd, end - start, start))
        out = []
        for index in range(count):
            start = int(offsets[index])
            end = start + int(lengths[index])
            if end <= start:
                out.append(b"")
                continue
            slot = int(np.searchsorted(starts, start, side="right")) - 1
            if slot < 0:
                out.append(b"")
                continue
            base = int(starts[slot])
            data = blobs[slot]
            if start < base or end > base + len(data):
                out.append(b"")
                continue
            out.append(data[start - base:end - base])
        return out

    def values_at(self, row_ids):
        """读取指定 row_id 的值列表。row_ids 为整数数组。"""
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if len(row_ids) == 0:
            return []
        ids, slots, positions = self._slots_for(row_ids)
        if len(ids) == 0:
            return [None] * len(row_ids)
        if self.is_text:
            texts = self._read_text(slots)
            decoded = [text.decode("utf-8", "replace") for text in texts]
            return [decoded[pos] for pos in positions]
        numbers = slots.astype(np.int64)
        return [int(numbers[pos]) for pos in positions]

    def compare_array(self, row_ids):
        """返回可直接做向量化比较的数组（VARCHAR 返回 bytes 列表）。"""
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if len(row_ids) == 0:
            return [] if self.is_text else np.zeros(0, dtype=np.int64)
        ids, slots, positions = self._slots_for(row_ids)
        if len(ids) == 0:
            return [b""] * len(row_ids) if self.is_text else np.zeros(len(row_ids), dtype=np.int64)
        if self.is_text:
            texts = self._read_text(slots)
            return [texts[pos] for pos in positions]
        numbers = slots.astype(np.int64)
        return numbers[positions]


class TableStorage:
    """单表的持久化存储。"""

    def __init__(self, root, name, schema):
        self.name = name
        self.schema = schema
        self.columns = schema["columns"]
        self.primary_key = schema["primary_key"]
        self.vector_column = schema.get("vector_column")
        self.vector_dim = schema.get("vector_dim") or 0
        self.directory = os.path.join(root, "tables", name)
        self.cols_dir = os.path.join(self.directory, "cols")
        self.index_dir = os.path.join(self.directory, "ivf")
        self.wal_path = os.path.join(self.directory, "wal.bin")
        self.vec_path = os.path.join(self.directory, "vec.f32")
        self.stores = {}
        self.wal_fd = None
        self.vec_fd = None
        self.wal_len = 0
        self.vec_len = 0
        self.next_row_id = 0
        self.live = {}          # pk -> row_id
        self.row_pk = {}        # row_id -> pk
        self.version = 0        # 数据版本号，用于缓存失效
        self._live_cache = None
        self._live_cache_version = -1
        self._vec_cache = None  # 小表时缓存向量，大表不缓存
        self._vec_cache_len = 0
        self._vec_cache_off = False

    # ---- 打开与恢复 ----

    def open(self, marks):
        os.makedirs(self.cols_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        for column in self.columns:
            if column["type"] == "vector":
                # 向量存放在 vec.f32，不建普通列文件
                continue
            store = ColumnStore(self.cols_dir, column["name"], column["type"])
            store.open()
            self.stores[column["name"]] = store
        self.wal_fd = os.open(self.wal_path, os.O_RDWR | os.O_CREAT, 0o644)
        self.wal_len = os.fstat(self.wal_fd).st_size
        self.vec_fd = os.open(self.vec_path, os.O_RDWR | os.O_CREAT, 0o644)
        self.vec_len = os.fstat(self.vec_fd).st_size
        marks = marks or {}
        # 截断到最后一个提交点，未提交的数据直接丢弃
        wal_mark = marks.get("wal_len")
        if wal_mark is not None and wal_mark < self.wal_len:
            os.ftruncate(self.wal_fd, wal_mark)
            self.wal_len = wal_mark
        vec_mark = marks.get("vec_len")
        if vec_mark is not None and vec_mark < self.vec_len:
            os.ftruncate(self.vec_fd, vec_mark)
            self.vec_len = vec_mark
        for name, column_marks in (marks.get("cols") or {}).items():
            store = self.stores.get(name)
            if store is not None:
                store.truncate(column_marks[0], column_marks[1])
        self._replay()
        self._refresh_vector_cache()

    def _replay(self):
        """流式重放 WAL。

        逐块读取而不是整段载入内存；遇到半截记录（崩溃导致的撕裂写）即停止，
        该记录及其之后的记录都视为未提交。
        """
        if self.wal_len == 0:
            return
        buffer = b""
        base = 0     # buffer[0] 对应的文件偏移
        pos = 0      # 当前解析位置（相对 buffer）
        header = LENGTH_PREFIX.size
        while True:
            while len(buffer) - pos < header:
                more = os.pread(self.wal_fd, READ_CHUNK, base + len(buffer))
                if not more:
                    break
                buffer += more
            if len(buffer) - pos < header:
                break
            size = LENGTH_PREFIX.unpack(buffer[pos:pos + header])[0]
            while len(buffer) - pos < header + size:
                more = os.pread(self.wal_fd, READ_CHUNK, base + len(buffer))
                if not more:
                    break
                buffer += more
            if len(buffer) - pos < header + size:
                break
            payload = buffer[pos + header:pos + header + size]
            try:
                record = json.loads(payload.decode("utf-8"))
            except (ValueError, UnicodeError):
                break
            row_id = record["row_id"]
            pk = record["pk"]
            self.next_row_id = max(self.next_row_id, row_id + 1)
            if record.get("deleted"):
                self.live.pop(pk, None)
                self.row_pk.pop(row_id, None)
            else:
                self.live[pk] = row_id
                self.row_pk[row_id] = pk
            pos += header + size
            if pos >= READ_CHUNK:
                buffer = buffer[pos:]
                base += pos
                pos = 0
        self.version += 1
        self._live_cache = None

    # ---- 提交点与回滚 ----

    def checkpoint(self):
        cols = {}
        for name, store in self.stores.items():
            cols[name] = (store.data_len, store.str_len)
        return {"wal_len": self.wal_len, "vec_len": self.vec_len,
                "next_row_id": self.next_row_id, "cols": cols}

    def restore(self, checkpoint):
        """回滚到检查点：截断文件后重放 WAL（幂等，可重复执行）。"""
        self.wal_len = checkpoint["wal_len"]
        self.vec_len = checkpoint["vec_len"]
        self.next_row_id = checkpoint["next_row_id"]
        os.ftruncate(self.wal_fd, self.wal_len)
        os.ftruncate(self.vec_fd, self.vec_len)
        for name, lengths in checkpoint["cols"].items():
            store = self.stores.get(name)
            if store is not None:
                store.truncate(lengths[0], lengths[1])
        self.live = {}
        self.row_pk = {}
        self._replay()
        self.version += 1
        self._live_cache = None
        self._refresh_vector_cache()

    # ---- 写入 ----

    def allocate_row_id(self):
        row_id = self.next_row_id
        self.next_row_id += 1
        return row_id

    def write_vector(self, row_id, values):
        if self.vector_dim <= 0 or values is None:
            return
        array = np.asarray(values, dtype="<f4")
        payload = array.tobytes()
        offset = row_id * self.vector_dim * 4
        os.pwrite(self.vec_fd, payload, offset)
        target = offset + len(payload)
        if target > self.vec_len:
            current = os.fstat(self.vec_fd).st_size
            if target > current:
                os.pwrite(self.vec_fd, b"\x00" * (target - current), current)
            self.vec_len = target
        self._cache_vector(row_id, array)

    def append_wal(self, row_id, pk, deleted):
        record = {"row_id": row_id, "pk": pk, "deleted": deleted}
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
        os.pwrite(self.wal_fd, LENGTH_PREFIX.pack(len(payload)), self.wal_len)
        os.pwrite(self.wal_fd, payload, self.wal_len + LENGTH_PREFIX.size)
        self.wal_len += LENGTH_PREFIX.size + len(payload)

    def write_row(self, row_id, pk, values):
        for column in self.columns:
            if column["type"] == "vector":
                # 向量单独存放在 vec.f32，不进普通列槽位
                continue
            store = self.stores[column["name"]]
            value = values.get(column["name"])
            if value is None:
                value = "" if column["type"] == "varchar" else 0
            store.write(row_id, value)
        self.append_wal(row_id, pk, False)
        self.live[pk] = row_id
        self.row_pk[row_id] = pk
        self.version += 1
        self._live_cache = None

    def delete_row(self, row_id, pk):
        self.append_wal(row_id, pk, True)
        self.live.pop(pk, None)
        self.row_pk.pop(row_id, None)
        self.version += 1
        self._live_cache = None

    # ---- 读取 ----

    def live_row_ids(self):
        """当前存活行的 row_id（升序）。带版本缓存，避免重复排序。"""
        if self._live_cache is None or self._live_cache_version != self.version:
            self._live_cache = np.array(sorted(self.live.values()), dtype=np.int64)
            self._live_cache_version = self.version
        return self._live_cache

    def live_pks(self):
        return np.array(sorted(self.live.keys()), dtype=np.int64)

    def column_values(self, name, row_ids):
        if name == self.vector_column:
            return self.vector_values(row_ids)
        return self.stores[name].values_at(row_ids)

    def vector_values(self, row_ids):
        """返回向量列的取值：JSON 数组形式的 float32 列表。

        按传入 row_ids 的顺序返回，逐块读取，避免一次持有整表向量。
        """
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if self.vector_dim <= 0 or len(row_ids) == 0:
            return []
        out = []
        for begin in range(0, len(row_ids), CHUNK):
            window = row_ids[begin:begin + CHUNK]
            ids, block = self.read_vectors(window)
            if len(ids) == 0:
                out.extend([[0.0] * self.vector_dim for _ in range(len(window))])
                continue
            positions = np.searchsorted(ids, window)
            positions = np.clip(positions, 0, len(ids) - 1)
            out.extend([[float(x) for x in row] for row in block[positions].tolist()])
        return out

    def primary_key_values(self, row_ids):
        """返回指定 row_id 对应的主键（int64 数组）。"""
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if len(row_ids) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.asarray(self.stores[self.primary_key].compare_array(row_ids),
                          dtype=np.int64)

    def column_compare(self, name, row_ids):
        return self.stores[name].compare_array(row_ids)

    def read_vector(self, row_id):
        if self.vector_dim <= 0:
            return None
        offset = row_id * self.vector_dim * 4
        raw = os.pread(self.vec_fd, self.vector_dim * 4, offset)
        if len(raw) < self.vector_dim * 4:
            return np.zeros(self.vector_dim, dtype=np.float32)
        return np.frombuffer(raw, dtype="<f4")

    def _vector_runs(self, ids):
        """把递增的 row_id 切成连续段（允许小空洞），返回 (low, high, 切片)。"""
        runs = []
        begin = 0
        for index in range(1, len(ids)):
            if int(ids[index]) - int(ids[index - 1]) > GAP_THRESHOLD:
                runs.append((int(ids[begin]), int(ids[index - 1]), slice(begin, index)))
                begin = index
        runs.append((int(ids[begin]), int(ids[-1]), slice(begin, len(ids))))
        return runs

    def _refresh_vector_cache(self):
        """在内存预算之内缓存向量；超出预算则不缓存，改走磁盘分块读取。"""
        self._vec_cache = None
        self._vec_cache_len = 0
        self._vec_cache_off = False
        if self.vector_dim <= 0:
            return
        count = self.vec_len // 4 // self.vector_dim
        if count * self.vector_dim * 4 > VECTOR_CACHE_LIMIT:
            self._vec_cache_off = True
            return
        raw = os.pread(self.vec_fd, count * self.vector_dim * 4, 0)
        if len(raw) < count * self.vector_dim * 4:
            self._vec_cache_off = True
            return
        self._vec_cache = np.frombuffer(raw, dtype="<f4").reshape(count, self.vector_dim).copy()
        self._vec_cache_len = count

    def _cache_vector(self, row_id, values):
        """把刚写入的向量同步进内存缓存（超出预算则放弃缓存）。"""
        if self.vector_dim <= 0 or self._vec_cache_off:
            return
        needed = (row_id + 1) * self.vector_dim * 4
        if needed > VECTOR_CACHE_LIMIT:
            self._vec_cache = None
            self._vec_cache_len = 0
            self._vec_cache_off = True
            return
        if self._vec_cache is None:
            self._vec_cache = np.zeros((0, self.vector_dim), dtype=np.float32)
        if row_id >= len(self._vec_cache):
            grow = max(row_id + 1, max(1024, len(self._vec_cache) * 2))
            if grow * self.vector_dim * 4 > VECTOR_CACHE_LIMIT:
                grow = row_id + 1
                if grow * self.vector_dim * 4 > VECTOR_CACHE_LIMIT:
                    self._vec_cache = None
                    self._vec_cache_len = 0
                    self._vec_cache_off = True
                    return
            block = np.zeros((grow, self.vector_dim), dtype=np.float32)
            block[:len(self._vec_cache)] = self._vec_cache
            self._vec_cache = block
        self._vec_cache[row_id] = values
        self._vec_cache_len = max(self._vec_cache_len, row_id + 1)

    def iter_vector_batches(self, row_ids, size=CHUNK):
        """按块产出 (row_id 数组, 向量数组)。

        命中内存缓存时直接切片；否则按连续段从磁盘读，每次只驻留一块。
        """
        if self.vector_dim <= 0 or len(row_ids) == 0:
            return
        count = self.vec_len // 4 // self.vector_dim
        ids = np.unique(np.asarray(row_ids, dtype=np.int64))
        ids = ids[ids < count]
        if len(ids) == 0:
            return
        for begin in range(0, len(ids), size):
            window = ids[begin:begin + size]
            if self._vec_cache is not None and int(window[-1]) < len(self._vec_cache):
                yield window, self._vec_cache[window]
                continue
            for low, high, part in self._vector_runs(window):
                raw = os.pread(self.vec_fd, (high - low + 1) * self.vector_dim * 4,
                               low * self.vector_dim * 4)
                block = np.frombuffer(raw, dtype="<f4").reshape(-1, self.vector_dim)
                yield window[part], block[window[part] - low]

    def read_vectors(self, row_ids):
        """读取多个向量，返回 (row_id 数组, (n, dim) float32)。

        调用方应对 row_ids 分块传入，避免一次持有全表向量。
        """
        if self.vector_dim <= 0 or len(row_ids) == 0:
            return (np.zeros(0, dtype=np.int64),
                    np.zeros((0, max(self.vector_dim, 1)), dtype=np.float32))
        size = os.fstat(self.vec_fd).st_size
        count = min(self.vec_len, size) // 4 // self.vector_dim
        ids = np.unique(np.asarray(row_ids, dtype=np.int64))
        ids = ids[ids < count]
        if len(ids) == 0:
            return ids, np.zeros((0, self.vector_dim), dtype=np.float32)
        out = np.zeros((len(ids), self.vector_dim), dtype=np.float32)
        for low, high, window in self._vector_runs(ids):
            raw = os.pread(self.vec_fd, (high - low + 1) * self.vector_dim * 4,
                           low * self.vector_dim * 4)
            block = np.frombuffer(raw, dtype="<f4").reshape(-1, self.vector_dim)
            out[window] = block[ids[window] - low]
        return ids, out

    def vector_block(self, row_ids):
        """向量化读取多个向量，返回 (n, dim) float32 数组。

        保留旧接口，内部走分段读取。
        """
        _, block = self.read_vectors(row_ids)
        return block

    def flush(self):
        for fd in (self.wal_fd, self.vec_fd):
            if fd is not None:
                os.fsync(fd)
        for store in self.stores.values():
            store.flush()

    def close(self):
        for fd_name in ("wal_fd", "vec_fd"):
            fd = getattr(self, fd_name, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, fd_name, None)
        for store in self.stores.values():
            store.close()


class Database:
    """数据库目录管理：catalog + commit log + 多表存储。"""

    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.tables_dir = os.path.join(self.data_dir, "tables")
        self.catalog_path = os.path.join(self.data_dir, "catalog.json")
        self.commit_path = os.path.join(self.data_dir, "commit.log")
        self.tables = {}
        self.schemas = {}

    def open(self):
        os.makedirs(self.tables_dir, exist_ok=True)
        committed = self._read_commit_log()
        catalog = self._read_catalog()
        for name, schema in catalog.items():
            table = TableStorage(self.data_dir, name, schema)
            table.open(committed.get(name))
            self.tables[name] = table
            self.schemas[name] = schema

    def _read_catalog(self):
        if not os.path.exists(self.catalog_path):
            return {}
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return data.get("tables") or {}

    def _read_commit_log(self):
        """读取每个表最后一次提交的文件长度；截断的尾行视为未提交。

        兼容两种格式：
            旧格式：每行一个表 {"table": name, "wal_len": ...}
            新格式：每行一次提交 {"tables": {name: {...}}}（多表原子提交）
        """
        result = {}
        if not os.path.exists(self.commit_path):
            return result
        with open(self.commit_path, "rb") as handle:
            raw = handle.read()
        for line in raw.split(b"\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeError):
                continue
            tables = entry.get("tables")
            if isinstance(tables, dict):
                for name, mark in tables.items():
                    if isinstance(mark, dict):
                        result[name] = mark
                continue
            name = entry.get("table")
            if name:
                result[name] = entry
        return result

    def create_table(self, name, schema):
        table = TableStorage(self.data_dir, name, schema)
        table.open(None)
        self.tables[name] = table
        self.schemas[name] = schema
        self.write_catalog()
        return table

    def drop_table(self, name):
        table = self.tables.pop(name, None)
        self.schemas.pop(name, None)
        if table:
            table.close()
        self.write_catalog()

    def write_catalog(self):
        tmp = self.catalog_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"tables": self.schemas}, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.catalog_path)
        _fsync_dir(self.data_dir)

    def commit(self):
        """提交点：落盘所有表并向 commit.log 追加一行。

        一次提交只写一行，覆盖所有表的长度信息；因此多表事务要么整体生效，
        要么整体不生效，不会出现「一行写了一半」的中间状态。
        """
        for table in self.tables.values():
            table.flush()
        tables = {}
        for name, table in self.tables.items():
            cols = {}
            for column_name, store in table.stores.items():
                cols[column_name] = [store.data_len, store.str_len]
            tables[name] = {"wal_len": table.wal_len, "vec_len": table.vec_len,
                            "cols": cols}
        payload = json.dumps({"tables": tables}, ensure_ascii=False).encode("utf-8")
        with open(self.commit_path, "ab") as handle:
            handle.write(payload + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def close(self):
        for table in self.tables.values():
            table.close()
        self.tables.clear()

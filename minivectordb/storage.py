"""表存储层：列式存储 + WAL + 崩溃恢复 + 事务撤销。

磁盘布局（全部位于 /data 下）：

    /data/catalog.json                 表结构清单
    /data/commit.log                   提交点记录，每行一个 JSON
    /data/tables/<table>/wal.bin       行元信息日志（row_id / pk / 删除标记）
    /data/tables/<table>/cols/<c>.dat  列数据，槽位下标 = row_id，8 字节定长
    /data/tables/<table>/cols/<c>.str  VARCHAR 字符串堆（追加写）
    /data/tables/<table>/vec.f32       向量块，偏移 = row_id * dim * 4

设计要点：
    * 列为定长槽位，可直接 mmap 并做 numpy 向量化比较，避免逐行 JSON 解析。
    * 行版本 row_id 单调递增且稠密；更新一条记录即分配新 row_id 写新槽位，
      旧槽位自然失效，回滚时按提交点截断文件即可。
    * 向量以定长块存放，同样可 mmap 后向量化计算距离。
    * commit.log 记录每次提交时各文件的长度；启动时截断到最后一个提交点，
      因此「已确认提交不丢、未提交不部分生效」。
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
            payload = value.encode("utf-8")
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

    def values_at(self, row_ids):
        """读取指定 row_id 的值列表。row_ids 为 numpy 整数数组。"""
        if len(row_ids) == 0:
            return []
        if self.is_text:
            blob = self._string_blob()
            slots = self._slot_array()[row_ids]
            out = []
            for offset, length in zip(slots["off"], slots["len"]):
                out.append(blob[offset:offset + length].decode("utf-8"))
            return out
        raw = self._slot_array()[row_ids]
        return [int(x) for x in raw]

    def _slot_array(self):
        """返回槽位数组。

        不使用 memmap：文件扩展后已建立的 memmap 视图不会反映新写入的页，
        会读到补零前的旧内容。这里用 os.pread 读取，保证是文件当前的真实字节。
        """
        dtype = VARCHAR_SLOT if self.is_text else BIGINT_SLOT
        if self.data_fd is None:
            return np.zeros(0, dtype=dtype)
        size = os.fstat(self.data_fd).st_size
        count = min(self.data_len, size) // SLOT_WIDTH
        if count == 0:
            return np.zeros(0, dtype=dtype)
        raw = os.pread(self.data_fd, count * SLOT_WIDTH, 0)
        return np.frombuffer(raw, dtype=dtype, count=count)

    def _string_blob(self):
        if self.str_fd is None:
            return b""
        size = os.fstat(self.str_fd).st_size
        if size == 0:
            return b""
        return os.pread(self.str_fd, size, 0)

    def compare_array(self, row_ids):
        """返回可直接做向量化比较的数组（VARCHAR 返回 bytes 列表）。"""
        if self.is_text:
            blob = self._string_blob()
            slots = self._slot_array()[row_ids]
            return [blob[o:o + n] for o, n in zip(slots["off"], slots["len"])]
        return np.asarray(self._slot_array()[row_ids], dtype=np.int64)


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

    def _replay(self):
        if self.wal_len == 0:
            return
        raw = os.pread(self.wal_fd, self.wal_len, 0)
        offset = 0
        while offset + LENGTH_PREFIX.size <= self.wal_len:
            size = LENGTH_PREFIX.unpack(raw[offset:offset + LENGTH_PREFIX.size])[0]
            start = offset + LENGTH_PREFIX.size
            payload = raw[start:start + size]
            if len(payload) != size:
                break
            record = json.loads(payload.decode("utf-8"))
            row_id = record["row_id"]
            pk = record["pk"]
            self.next_row_id = max(self.next_row_id, row_id + 1)
            if record.get("deleted"):
                self.live.pop(pk, None)
                self.row_pk.pop(row_id, None)
            else:
                self.live[pk] = row_id
                self.row_pk[row_id] = pk
            offset = start + size

    # ---- 提交点与回滚 ----

    def checkpoint(self):
        cols = {}
        for name, store in self.stores.items():
            cols[name] = (store.data_len, store.str_len)
        return {"wal_len": self.wal_len, "vec_len": self.vec_len,
                "next_row_id": self.next_row_id, "cols": cols}

    def restore(self, checkpoint):
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

    def delete_row(self, row_id, pk):
        self.append_wal(row_id, pk, True)
        self.live.pop(pk, None)
        self.row_pk.pop(row_id, None)
        self.version += 1

    # ---- 读取 ----

    def live_row_ids(self):
        return np.fromiter(self.live.values(), dtype=np.int64, count=len(self.live))

    def live_pks(self):
        return np.fromiter(self.live.keys(), dtype=np.int64, count=len(self.live))

    def column_values(self, name, row_ids):
        return self.stores[name].values_at(row_ids)

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

    def vector_block(self, row_ids):
        """向量化读取多个向量，返回 (n, dim) float32 数组。

        用 os.pread 而非 memmap：文件扩展后 memmap 视图不会反映新写入的页。
        """
        if self.vector_dim <= 0 or len(row_ids) == 0:
            return np.zeros((0, max(self.vector_dim, 1)), dtype=np.float32)
        size = os.fstat(self.vec_fd).st_size
        count = min(self.vec_len, size) // 4 // self.vector_dim
        if count == 0:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        row_ids = np.asarray(row_ids, dtype=np.int64)
        row_ids = row_ids[row_ids < count]
        if len(row_ids) == 0:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        # 连续区间一次性读取，减少系统调用
        low = int(row_ids.min())
        high = int(row_ids.max())
        raw = os.pread(self.vec_fd, (high - low + 1) * self.vector_dim * 4,
                       low * self.vector_dim * 4)
        block = np.frombuffer(raw, dtype="<f4").reshape(-1, self.vector_dim)
        return block[row_ids - low].astype(np.float32, copy=True)

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
        """读取每个表最后一次提交的文件长度；截断的尾行视为未提交。"""
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
            table = entry.get("table")
            if table:
                result[table] = entry
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
        """提交点：落盘所有表并追加 commit.log。"""
        for table in self.tables.values():
            table.flush()
        entries = []
        for name, table in self.tables.items():
            cols = {}
            for column_name, store in table.stores.items():
                cols[column_name] = [store.data_len, store.str_len]
            entries.append({"table": name, "wal_len": table.wal_len,
                            "vec_len": table.vec_len, "cols": cols})
        with open(self.commit_path, "ab") as handle:
            for entry in entries:
                handle.write((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

    def close(self):
        for table in self.tables.values():
            table.close()
        self.tables.clear()

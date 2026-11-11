"""自研向量索引：IVF（倒排文件）近邻检索。

不使用 Faiss / VSAG / HNSWlib 等现成向量索引库。

结构：
    训练：对采样出的向量做 k-means（随机初始化 + Lloyd 迭代），得到 nlist 个聚类中心
    分配：把全表向量分块扫描一遍，每条向量归属到最近的聚类中心
    检索：先探测 nprobe 个最相近的聚类，收集候选，再精确计算距离取 TopK

内存策略：
    索引本身只保存 centroids / assignments / row_ids，不保留全量向量；
    建索引与检索时的向量都按块从表存储读取，因此数据量可以超过内存预算。

召回率保障：
    默认 nlist = sqrt(N)、nprobe = 约半数聚类，候选集远大于 K；
    存在等值过滤时，额外把候选行所属的全部聚类纳入探测范围，
    保证过滤后的真值集合不会被漏掉。
"""

import json
import os

import numpy as np

# 一次读入的最大向量条数
CHUNK = 4096
# 训练样本上限：k-means 只在样本上进行，避免把全表向量放进内存
TRAIN_MAX = 20000
# 索引可缓存的向量字节上限：超过这个预算就不缓存，走磁盘分块读取
CACHE_LIMIT = 64 << 20


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


class VectorIndex:
    def __init__(self, dim):
        self.dim = dim
        self.nlist = 0
        self.nprobe = 0
        self.centroids = None      # (nlist, dim) float32
        self.row_ids = None        # (N,) int64，向量下标 -> row_id
        self.pks = None            # (N,) int64，向量下标 -> 主键
        self.assignments = None    # (N,) int32，每个向量所属聚类
        self.dirty = True
        self.signature = None      # 建索引时的数据指纹，用于判断是否过期

    # ---- 训练与构建 ----

    def _kmeans(self, sample, nlist, seed):
        count = sample.shape[0]
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(count, size=nlist, replace=False)
        centroids = sample[init_idx].astype(np.float64).copy()
        for _ in range(12):
            distances = (sample ** 2).sum(axis=1)[:, None] \
                - 2.0 * sample.dot(centroids.T) + (centroids ** 2).sum(axis=1)[None, :]
            assignments = np.argmin(distances, axis=1)
            for cluster in range(nlist):
                members = sample[assignments == cluster]
                if members.shape[0]:
                    centroids[cluster] = members.mean(axis=0)
        return centroids.astype(np.float32)

    def build(self, table, row_ids, nlist=None, nprobe=None, seed=12345):
        """用表存储构建索引。

        row_ids 为全部存活行的 row_id（升序）。构建过程分块读取向量，
        内存中只保留训练样本与分配结果。
        """
        row_ids = np.asarray(row_ids, dtype=np.int64)
        count = len(row_ids)
        self.row_ids = row_ids
        self.pks = table.primary_key_values(row_ids)
        if count == 0:
            self.nlist = 0
            self.centroids = None
            self.assignments = np.zeros(0, dtype=np.int32)
            self.dirty = False
            self.signature = self._signature(table)
            return
        # 1) 采样训练
        if count <= TRAIN_MAX:
            train_ids = row_ids
        else:
            step = count / float(TRAIN_MAX)
            picks = np.floor(np.arange(TRAIN_MAX) * step).astype(np.int64)
            train_ids = row_ids[np.unique(picks)]
        samples = []
        for ids, vectors in table.iter_vector_batches(train_ids):
            samples.append(vectors)
        sample = np.concatenate(samples) if samples else np.zeros((0, self.dim), dtype=np.float32)
        if sample.shape[0] == 0:
            self.nlist = 0
            self.centroids = None
            self.assignments = np.zeros(count, dtype=np.int32)
            self.dirty = False
            self.signature = self._signature(table)
            return
        if nlist is None:
            nlist = max(1, min(int(np.sqrt(count)), 4096))
        nlist = max(1, min(nlist, sample.shape[0]))
        self.centroids = self._kmeans(sample.astype(np.float64), nlist, seed)
        self.nlist = nlist
        # 2) 分块分配
        assignments = np.zeros(count, dtype=np.int32)
        centroids = self.centroids.astype(np.float64)
        centroid_norms = (centroids ** 2).sum(axis=1)
        for ids, vectors in table.iter_vector_batches(row_ids):
            if len(ids) == 0:
                continue
            positions = np.searchsorted(row_ids, ids)
            block = vectors.astype(np.float64)
            distances = (block ** 2).sum(axis=1)[:, None] - 2.0 * block.dot(centroids.T) \
                + centroid_norms[None, :]
            assignments[positions] = np.argmin(distances, axis=1).astype(np.int32)
        self.assignments = assignments
        if nprobe is None:
            nprobe = max(1, min(nlist, int(np.ceil(nlist * 0.5)) + 1))
        self.nprobe = nprobe
        self.dirty = False
        self.signature = self._signature(table)

    @staticmethod
    def _signature(table):
        """数据指纹：行数与 row_id 之和，足以反映插入/更新/删除。"""
        row_ids = table.live_row_ids()
        return [int(len(row_ids)), int(row_ids.sum())]

    def is_current(self, table):
        if self.dirty or self.row_ids is None or len(self.row_ids) == 0:
            return False
        return self.signature == self._signature(table)

    # ---- 检索 ----

    def search(self, table, query, k, allowed=None):
        """返回 [(row_id, distance)]，按距离升序、同距按 row_id 升序。

        allowed 为允许参与检索的 row_id 数组；None 表示不限制。
        向量按块从表存储读取，内存中不保留全量向量。
        """
        query = np.asarray(query, dtype=np.float64)
        if self.row_ids is None or len(self.row_ids) == 0:
            return []
        positions = np.arange(len(self.row_ids), dtype=np.int64)
        if allowed is not None:
            allowed = np.asarray(allowed, dtype=np.int64)
            positions = positions[np.isin(self.row_ids, allowed)]
        if len(positions) == 0:
            return []
        if self.centroids is None or self.nlist <= 1:
            candidates = positions
        else:
            centroids = self.centroids.astype(np.float64)
            cd = (centroids ** 2).sum(axis=1) - 2.0 * centroids.dot(query) \
                + float((query ** 2).sum())
            probe_order = np.argsort(cd, kind="stable")
            probe_count = max(1, min(self.nprobe, self.nlist))
            selected = {int(cluster) for cluster in probe_order[:probe_count]}
            # 过滤后候选可能只落在少数聚类，逐一纳入以免漏召回
            for cluster in np.unique(self.assignments[positions]):
                selected.add(int(cluster))
            mask = np.isin(self.assignments, np.array(sorted(selected), dtype=np.int64))
            candidates = positions[mask[positions]]
        if len(candidates) == 0:
            return []
        candidate_ids = self.row_ids[candidates]
        best_dist = np.zeros(0, dtype=np.float64)
        best_pk = np.zeros(0, dtype=np.int64)
        best_ids = np.zeros(0, dtype=np.int64)
        for ids, vectors in table.iter_vector_batches(candidate_ids):
            if len(ids) == 0:
                continue
            diff = vectors.astype(np.float64) - query
            distances = np.sqrt((diff * diff).sum(axis=1))
            pks = table.primary_key_values(ids)
            best_dist = np.concatenate([best_dist, distances])
            best_pk = np.concatenate([best_pk, pks])
            best_ids = np.concatenate([best_ids, ids])
            if len(best_dist) > 8 * k + CHUNK:
                order = np.lexsort((best_pk, best_dist))[:k]
                best_dist, best_pk, best_ids = best_dist[order], best_pk[order], best_ids[order]
        order = np.lexsort((best_pk, best_dist))[:k]
        return [(int(best_ids[i]), float(best_dist[i])) for i in order]

    # ---- 持久化 ----

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "centroids.npy"),
                self.centroids if self.centroids is not None
                else np.zeros((0, self.dim), dtype=np.float32))
        np.save(os.path.join(directory, "assignments.npy"),
                self.assignments if self.assignments is not None
                else np.zeros(0, dtype=np.int32))
        np.save(os.path.join(directory, "row_ids.npy"),
                self.row_ids if self.row_ids is not None
                else np.zeros(0, dtype=np.int64))
        with open(os.path.join(directory, "meta.txt"), "w", encoding="utf-8") as handle:
            handle.write("dim=%d\nnlist=%d\nnprobe=%d\n" % (self.dim, self.nlist, self.nprobe))
            handle.write("signature=%s\n" % json.dumps(self.signature))

    def load(self, directory):
        try:
            self.centroids = np.load(os.path.join(directory, "centroids.npy"))
            self.assignments = np.load(os.path.join(directory, "assignments.npy"))
            self.row_ids = np.load(os.path.join(directory, "row_ids.npy"))
        except (OSError, ValueError):
            return False
        nlist = 0
        nprobe = 0
        signature = None
        try:
            with open(os.path.join(directory, "meta.txt"), "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("nlist="):
                        nlist = int(line[6:].strip())
                    elif line.startswith("nprobe="):
                        nprobe = int(line[7:].strip())
                    elif line.startswith("signature="):
                        signature = json.loads(line[10:].strip())
        except (OSError, ValueError):
            pass
        self.nlist = nlist
        self.nprobe = nprobe
        self.signature = signature
        self.dirty = False
        return True

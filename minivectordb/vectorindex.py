"""自研向量索引：IVF（倒排文件）近邻检索。

不使用 Faiss / VSAG / HNSWlib 等现成向量索引库。

结构：
    训练：对样本向量做 k-means（k-means++ 初始化 + Lloyd 迭代），得到 nlist 个聚类中心
    分配：每条向量归属到最近的聚类中心
    检索：先探测 nprobe 个最相近的聚类，收集候选，再精确计算距离取 TopK

召回率保障：
    默认 nlist = sqrt(N)、nprobe = 约半数聚类，候选集远大于 K；
    存在等值过滤时，额外把候选行所属的全部聚类纳入探测范围，
    保证过滤后的真值集合不会被漏掉。
"""

import os

import numpy as np


class VectorIndex:
    def __init__(self, dim):
        self.dim = dim
        self.nlist = 0
        self.nprobe = 0
        self.centroids = None
        self.vectors = None          # (N, dim) float32
        self.row_ids = []            # 向量下标 -> row_id
        self.assignments = None      # 每个向量所属聚类
        self.dirty = True

    # ---- 训练与构建 ----

    def build(self, vectors, row_ids, nlist=None, nprobe=None, seed=12345):
        vectors = np.asarray(vectors, dtype=np.float32)
        count = vectors.shape[0]
        if count == 0:
            self.nlist = 0
            self.centroids = None
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)
            self.row_ids = []
            self.assignments = np.zeros(0, dtype=np.int32)
            self.dirty = False
            return
        if nlist is None:
            nlist = max(1, min(int(np.sqrt(count)), 4096))
        nlist = max(1, min(nlist, count))
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(count, size=nlist, replace=False)
        centroids = vectors[init_idx].astype(np.float32).copy()
        assignments = np.zeros(count, dtype=np.int32)
        centroid_norms = (centroids ** 2).sum(axis=1)
        for _ in range(12):
            chunk = max(1, min(count, 65536 // max(1, nlist)))
            for begin in range(0, count, chunk):
                end = min(count, begin + chunk)
                block = vectors[begin:end]
                distances = (block ** 2).sum(axis=1)[:, None] - 2.0 * block.dot(centroids.T) \
                    + centroid_norms[None, :]
                assignments[begin:end] = np.argmin(distances, axis=1).astype(np.int32)
            for cluster in range(nlist):
                members = vectors[assignments == cluster]
                if members.shape[0]:
                    centroids[cluster] = members.mean(axis=0)
            centroid_norms = (centroids ** 2).sum(axis=1)
        self.nlist = nlist
        self.centroids = centroids.astype(np.float32)
        self.vectors = vectors
        self.row_ids = list(row_ids)
        self.assignments = assignments
        if nprobe is None:
            nprobe = max(1, min(nlist, int(np.ceil(nlist * 0.5)) + 1))
        self.nprobe = nprobe
        self.dirty = False

    # ---- 检索 ----

    def search(self, query, k, allowed=None):
        """返回 [(row_id, distance)]，按距离升序、同距按 row_id 升序。

        allowed 为允许参与检索的 row_id 集合；None 表示不限制。
        """
        query = np.asarray(query, dtype=np.float32)
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []
        count = self.vectors.shape[0]
        if allowed is not None:
            candidates = np.array([i for i, rid in enumerate(self.row_ids)
                                   if rid in allowed], dtype=np.int64)
            if candidates.size == 0:
                return []
        else:
            candidates = None

        if self.centroids is None or self.nlist <= 1:
            indices = np.arange(count) if candidates is None else candidates
        else:
            cd = (self.centroids ** 2).sum(axis=1) - 2.0 * self.centroids.dot(query) \
                + float((query ** 2).sum())
            probe_order = np.argsort(cd, kind="stable")
            probe_count = max(1, min(self.nprobe, self.nlist))
            selected = {int(cluster) for cluster in probe_order[:probe_count]}
            if candidates is not None:
                # 过滤后候选可能只落在少数聚类，逐一纳入以免漏召回
                for cluster in set(self.assignments[candidates].tolist()):
                    selected.add(int(cluster))
            mask = np.isin(self.assignments, list(selected))
            indices = np.nonzero(mask)[0]
            if candidates is not None:
                indices = np.intersect1d(indices, candidates)

        if indices.size == 0:
            return []
        block = self.vectors[indices]
        diff = block - query
        distances = np.sqrt((diff * diff).sum(axis=1)).astype(np.float64)
        order = np.lexsort((np.asarray(self.row_ids)[indices], distances))
        picked = order[:k]
        return [(self.row_ids[indices[i]], float(distances[i])) for i in picked]

    # ---- 持久化 ----

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "centroids.npy"),
                self.centroids if self.centroids is not None
                else np.zeros((0, self.dim), dtype=np.float32))
        np.save(os.path.join(directory, "vectors.npy"),
                self.vectors if self.vectors is not None
                else np.zeros((0, self.dim), dtype=np.float32))
        np.save(os.path.join(directory, "assignments.npy"),
                self.assignments if self.assignments is not None
                else np.zeros(0, dtype=np.int32))
        with open(os.path.join(directory, "meta.txt"), "w", encoding="utf-8") as handle:
            handle.write("dim=%d\nnlist=%d\nnprobe=%d\n" % (self.dim, self.nlist, self.nprobe))
            for row_id in self.row_ids:
                handle.write("row=%d\n" % row_id)

    def load(self, directory):
        try:
            self.centroids = np.load(os.path.join(directory, "centroids.npy"))
            self.vectors = np.load(os.path.join(directory, "vectors.npy"))
            self.assignments = np.load(os.path.join(directory, "assignments.npy"))
        except (OSError, ValueError):
            return False
        row_ids = []
        nlist = 0
        nprobe = 0
        try:
            with open(os.path.join(directory, "meta.txt"), "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("row="):
                        row_ids.append(int(line[4:].strip()))
                    elif line.startswith("nlist="):
                        nlist = int(line[6:].strip())
                    elif line.startswith("nprobe="):
                        nprobe = int(line[7:].strip())
        except OSError:
            pass
        self.row_ids = row_ids
        self.nlist = nlist
        self.nprobe = nprobe
        self.dirty = False
        return True

#!/usr/bin/env python3
"""性能与召回率自测：1 万条 x 128 维写入 + 1000 条 TopK 查询。

对照题面 3.3 的基准口径：
    默认 10,000 条记录、128 维向量、1,000 条 TopK 查询，K = 10，
    其中一半查询带普通列等值过滤。无过滤和有过滤两组平均 Recall@10 均须 >= 0.95。

用法：
    python3 bench/bench.py [--rows 10000] [--dim 128] [--queries 1000] [--k 10]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minivectordb.engine import Engine  # noqa: E402

import numpy as np  # noqa: E402


def build_dataset(rows, dim, seed=20261111):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((rows, dim)).astype(np.float32)
    categories = rng.integers(0, 10, size=rows)
    return vectors, categories


def vector_literal(vector):
    """用 repr 输出 float32 的精确值，避免文本截断造成距离排序偏差。"""
    return "[" + ",".join(repr(float(v)) for v in vector.tolist()) + "]"


def vector_literal_query(vector):
    return "[" + ",".join(repr(float(v)) for v in vector.tolist()) + "]"


def run(rows, dim, queries, k, seed):
    vectors, categories = build_dataset(rows, dim)
    data_dir = tempfile.mkdtemp(prefix="minivectordb-bench-")
    try:
        engine = Engine(data_dir)
        engine.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category BIGINT, "
                       "content VARCHAR(32), embedding VECTOR(%d));" % dim)
        # 写入：每批 100 条并提交（对齐题面批量写入口径）
        write_start = time.monotonic()
        batch = 100
        for begin in range(0, rows, batch):
            chunk = vectors[begin:begin + batch]
            values = []
            for offset, vector in enumerate(chunk):
                row_id = begin + offset + 1
                cat = int(categories[begin + offset])
                values.append("(%d, %d, 'row-%d', '%s')" % (
                    row_id, cat, row_id, vector_literal(vector)))
            sql = "INSERT INTO docs VALUES " + ", ".join(values) + ";"
            result = engine.execute(sql)
            if result.get("affected_rows") != len(values):
                raise SystemExit("insert failed at %d" % begin)
        write_elapsed = time.monotonic() - write_start

        # 建立近似索引
        index_start = time.monotonic()
        engine.execute("CREATE VECTOR INDEX docs_vec ON docs(embedding);")
        index_elapsed = time.monotonic() - index_start

        # 查询真值：从存储读回的向量计算全量精确 TopK，
        # 与判题器“真值在满足过滤条件的集合中计算”口径一致。
        rng = np.random.default_rng(seed + 1)
        query_vectors = rng.standard_normal((queries, dim)).astype(np.float32)
        half = queries // 2
        table = engine.db.tables["docs"]
        stored = np.stack([table.read_vector(table.live[pk])
                           for pk in sorted(table.live)]).astype(np.float32)
        stored_ids = np.array(sorted(table.live), dtype=np.int64)
        stored_cats = categories[stored_ids - 1]
        truth_nofilter = []
        truth_filter = []
        for i in range(queries):
            query = query_vectors[i]
            diff = stored - query
            distances = np.sqrt((diff * diff).sum(axis=1))
            if i < half:
                order = np.lexsort((stored_ids, distances))[:k]
                truth_nofilter.append(set(stored_ids[order].tolist()))
            else:
                target = int(categories[i])
                mask = stored_cats == target
                sub = np.nonzero(mask)[0]
                sub_dist = distances[sub]
                order = np.lexsort((stored_ids[sub], sub_dist))[:k]
                truth_filter.append(set(stored_ids[sub[order]].tolist()))

        # 实际查询（ann 模式）
        recalls_nofilter = []
        recalls_filter = []
        query_elapsed = 0.0
        for i in range(queries):
            query = query_vectors[i]
            if i < half:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance "
                       "FROM docs ORDER BY distance LIMIT %d;"
                       % (vector_literal(query), k))
                truth = truth_nofilter[i]
                bucket = recalls_nofilter
            else:
                target = int(categories[i])
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance "
                       "FROM docs WHERE category = %d ORDER BY distance LIMIT %d;"
                       % (vector_literal(query), target, k))
                truth = truth_filter[i - half]
                bucket = recalls_filter
            start = time.monotonic()
            result = engine.execute(sql, "ann")
            query_elapsed += time.monotonic() - start
            got = {row[0] for row in result["rows"]}
            if not truth:
                continue
            bucket.append(len(got & truth) / len(truth))

        avg_nofilter = sum(recalls_nofilter) / max(1, len(recalls_nofilter))
        avg_filter = sum(recalls_filter) / max(1, len(recalls_filter))
        report = {
            "rows": rows, "dim": dim, "queries": queries, "k": k,
            "write_seconds": round(write_elapsed, 3),
            "index_seconds": round(index_elapsed, 3),
            "query_seconds": round(query_elapsed, 3),
            "recall_at_k_nofilter": round(avg_nofilter, 4),
            "recall_at_k_filter": round(avg_filter, 4),
            "recall_at_k_mean": round((avg_nofilter + avg_filter) / 2, 4),
            "threshold": 0.95,
            "passed": avg_nofilter >= 0.95 and avg_filter >= 0.95,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    sys.exit(run(args.rows, args.dim, args.queries, args.k, args.seed))


if __name__ == "__main__":
    main()

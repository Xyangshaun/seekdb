# MiniVectorDB

轻量向量数据库，实现 OceanBase 数据库大赛 600081「轻量向量数据库（MiniVectorDB）」的赛题要求。

## 1. 语言与运行环境

| 项 | 值 |
| --- | --- |
| 语言 | Python 3（开发与验证使用 CPython 3.9.6） |
| 运行时依赖 | `numpy`（仅用于向量数组运算），版本锁定见 `requirements.txt` |
| 基础镜像 | `ubuntu:24.04` |
| 运行平台 | Linux x86_64 容器 |
| 数据目录 | `/data`（环境变量 `VECTORDB_DATA_DIR` 可覆盖，默认 `/data`） |

存储引擎、SQL 执行、向量索引与事务核心均为自行实现，未使用 SQLite / RocksDB 等存储引擎，也未使用 Faiss / VSAG / HNSWlib 等向量索引库。

## 2. 镜像构建与启动入口

```bash
docker build -t minivectordb:latest .
```

前台启动并保持标准输入打开（判题器通过进程管道连接 stdin/stdout）：

```bash
docker run -i --network=none \
  --mount type=bind,src=/local/data,dst=/data \
  minivectordb:latest
```

不使用 `-d`，不分配 `-t`，不暴露或映射端口。容器默认入口即 `python3 -m minivectordb --data-dir /data`。

本地开发可直接运行，无需构建镜像：

```bash
python3 -m minivectordb --data-dir /tmp/vectordb-data
```

## 3. 通信协议

遵循仓库根目录 `STDIO_PROTOCOL.md`：

- 启动时先打开 `/data`、完成恢复与索引重建，再输出 `{"type":"ready"}` 并 flush，之后才等待第一条请求。
- stdin 每行一个 JSON 请求：`{"id":N,"op":"execute","sql":"...","search_mode":"exact"}`；`search_mode` 取 `exact` 或 `ann`。
- stdout 每行一个 JSON 响应，原样回传 `id`（含其 JSON 类型）：
  - 查询成功：`{"id":N,"ok":true,"columns":[...],"rows":[[...]]}`
  - 修改成功：`{"id":N,"ok":true,"affected_rows":N}`
  - 失败：`{"id":N,"ok":false,"error":{"code":"...","message":"..."}}`
- `op":"shutdown"`：回滚未提交事务、落盘、返回 `{"id":N,"ok":true}`，随后以退出码 0 结束，关闭响应后不再有任何 stdout 输出。
- stdout 只输出协议报文；所有日志写入 stderr。

## 4. SQL 支持范围

支持：

- `CREATE TABLE`：`BIGINT`、`VARCHAR(n)`、`VECTOR(n)`；每表一个 BIGINT 主键，至多一个向量列。
- `CREATE VECTOR INDEX idx ON tbl(vec_col)`。
- `INSERT`（含多行）、`SELECT`（含 `SELECT *`）、`UPDATE`、`DELETE`。
- 条件查询：`=`、`<`、`<=`、`>`、`>=`、`AND`、`ORDER BY`（`ASC`/`DESC`）、`LIMIT`。
- 向量检索：`L2_DISTANCE(vec_col, '[...]')`、精确 TopK、带普通列等值过滤的 TopK。
- 事务：`BEGIN`、`COMMIT`、`ROLLBACK`。

不支持（返回 `error.code = "unsupported_sql"`）：`NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、外键、嵌套事务、主键更新。

语义要点：

- 距离为欧氏距离（非平方），按 float32 存储后的向量计算。
- 过滤后的 TopK 在满足 `WHERE` 的集合内计算，不做「先取 K 再过滤」。
- 距离相同时按主键升序；普通 `ORDER BY` 并列项同样按主键升序。
- `exact` 模式始终返回精确结果，即使已建立近似索引；`ann` 模式对向量 TopK 使用近似索引。
- `VARCHAR` 长度按 Unicode 字符数计算，字符串比较按 Unicode 码点顺序。

### 错误码

| code | 含义 |
| --- | --- |
| `duplicate_key` | 主键重复 |
| `dimension_mismatch` | 向量维度与列定义不符 |
| `unsupported_sql` | 不支持的语句或语法 |
| `no_such_table` / `no_such_column` | 表或列不存在 |
| `table_exists` | 建表时表已存在 |
| `value_too_long` | 超出 `VARCHAR(n)` 长度 |
| `invalid_vector` | 向量非法（NaN、Infinity、无法表示为 float32） |
| `invalid_value` / `type_mismatch` | 取值非法或类型不匹配 |
| `invalid_request` | 请求格式非法 |
| `internal_error` | 兜底错误（进程不会因此终止） |

## 5. 从空数据目录开始的完整操作示例

```bash
rm -rf /tmp/vectordb-demo && mkdir -p /tmp/vectordb-demo
docker run -i --rm --network=none \
  --mount type=bind,src=/tmp/vectordb-demo,dst=/data \
  minivectordb:latest <<'JSONL'
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), content VARCHAR(16), embedding VECTOR(4));","search_mode":"exact"}
{"id":2,"op":"execute","sql":"INSERT INTO docs VALUES (1,'a','一','[0,0,1,0]'), (2,'a','二','[0,0,0,1]'), (3,'b','三','[1,0,0,0]');","search_mode":"exact"}
{"id":3,"op":"execute","sql":"SELECT id, content FROM docs ORDER BY id;","search_mode":"exact"}
{"id":4,"op":"execute","sql":"SELECT id, L2_DISTANCE(embedding, '[0,0,1,0]') AS distance FROM docs ORDER BY distance LIMIT 2;","search_mode":"exact"}
{"id":5,"op":"execute","sql":"SELECT id, L2_DISTANCE(embedding, '[0,0,1,0]') AS distance FROM docs WHERE category = 'a' ORDER BY distance LIMIT 2;","search_mode":"ann"}
{"id":6,"op":"execute","sql":"BEGIN;","search_mode":"exact"}
{"id":7,"op":"execute","sql":"DELETE FROM docs WHERE id = 3;","search_mode":"exact"}
{"id":8,"op":"execute","sql":"ROLLBACK;","search_mode":"exact"}
{"id":9,"op":"execute","sql":"SELECT id FROM docs ORDER BY id;","search_mode":"exact"}
{"id":10,"op":"shutdown"}
JSONL
```

预期输出（除 ready 行外）：

```text
{"id":1,"ok":true}
{"id":2,"ok":true,"affected_rows":3}
{"id":3,"ok":true,"columns":["id","content"],"rows":[[1,"一"],[2,"二"],[3,"三"]]}
{"id":4,"ok":true,"columns":["id","distance"],"rows":[[1,0.0],[2,1.0]]}
{"id":5,"ok":true,"columns":["id","distance"],"rows":[[1,0.0],[2,1.0]]}
{"id":6,"ok":true}
{"id":7,"ok":true,"affected_rows":1}
{"id":8,"ok":true}
{"id":9,"ok":true,"columns":["id"],"rows":[[1],[2],[3]]}
{"id":10,"ok":true}
```

重启复用同一数据目录（数据仍在）：

```bash
docker run -i --rm --network=none \
  --mount type=bind,src=/tmp/vectordb-demo,dst=/data \
  minivectordb:latest <<'JSONL'
{"id":1,"op":"execute","sql":"SELECT id, content FROM docs ORDER BY id;","search_mode":"exact"}
{"id":2,"op":"shutdown"}
JSONL
```

## 6. 设计记录

见 `docs/DESIGN.md`：记录布局、向量索引方案、事务处理、提交持久化与恢复流程、已实现功能与已知限制。

## 7. 本地自测

公开用例（题包自带 20 个）：

```bash
export VECTORDB_IMAGE='minivectordb:latest'
python3 protocol_check/run.py --image "$VECTORDB_IMAGE" --handshake-only --output "protocol-results/handshake-$(date +%Y%m%d-%H%M%S)"
python3 protocol_check/run.py --image "$VECTORDB_IMAGE" --output "protocol-results/public-$(date +%Y%m%d-%H%M%S)"
```

无 Docker 时可用 `--command` 直接跑本地进程（同一套断言，不含容器隔离）：

```bash
python3 protocol_check/run.py --command '["python3","-m","minivectordb"]' --output "protocol-results/public-$(date +%Y%m%d-%H%M%S)"
```

性能与召回率自测：

```bash
python3 bench/bench.py --rows 10000 --dim 128 --queries 1000 --k 10
```

公开用例通过不代表正式准入通过：正式用例共 29 个，组织方另保留 9 个用例，且性能与 Agent prompt 独立计分。

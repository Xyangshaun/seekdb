# MiniVectorDB 设计与验证记录

本文记录实现方案、验证过程与已知限制，供评审与后续维护查阅。

## 1. 总体结构

```
minivectordb/
  errors.py       错误类型与错误码映射
  sqlparser.py    SQL 子集词法与语法分析
  storage.py      列式存储、WAL、提交点与崩溃恢复
  vectorindex.py  自研 IVF 向量索引
  engine.py       SQL 执行引擎、事务与向量化查询
  stdio_server.py stdin/stdout JSONL 协议服务端
bench/bench.py    写入吞吐、查询耗时与 Recall@K 自测
```

进程入口：`python3 -m minivectordb --data-dir /data`。

## 2. 记录布局

全部数据文件位于 `/data`：

```
/data/catalog.json                    表结构清单（原子替换写入）
/data/commit.log                      提交点记录，每行一个 JSON
/data/tables/<table>/wal.bin          行元信息日志：row_id / 主键 / 删除标记
/data/tables/<table>/cols/<col>.dat   列数据，槽位下标 = row_id，8 字节定长
/data/tables/<table>/cols/<col>.str   VARCHAR 字符串堆（追加写）
/data/tables/<table>/vec.f32          向量块，偏移 = row_id * dim * 4
/data/tables/<table>/ivf/             近似索引（centroids / vectors / assignments / meta）
```

### 列式槽位

普通列按 `row_id` 定位到 8 字节定长槽位：

- `BIGINT` 槽位为 8 字节小端整数；
- `VARCHAR` 槽位为 `(offset:u4, length:u4)`，实际字节存在同目录的 `.str` 追加堆中。

行版本 `row_id` 单调递增且稠密。更新一条记录即分配新的 `row_id` 写新槽位，旧槽位自然失效；
删除只写一条删除标记。这样回滚只需把文件截断到提交点，不必做反向日志应用。

### 为什么不用 mmap

文件扩展后，已建立的 mmap 视图不会反映新写入的页，会读到补零前的旧内容（实测复现：
写入 row_id=3 后 mmap 读回该槽位为 0，而磁盘上是正确值）。因此所有读取统一使用
`os.pread`，写入统一使用 `os.pwrite`，全程无缓冲，不受 Python 缓冲层位置语义影响。

## 3. 向量索引方案

自研 IVF（倒排文件）索引，未使用 Faiss / VSAG / HNSWlib。

- **训练**：k-means++ 风格随机初始化 + Lloyd 迭代（12 轮），聚类数 `nlist = sqrt(N)`（上限 4096）。
- **分配**：每条向量归属到最近聚类中心，`assignments` 数组记录。
- **检索**：计算查询向量到各聚类中心的距离，取最近的 `nprobe` 个聚类（默认约为聚类数的一半）；
  收集这些聚类的全部向量后精确计算 L2 距离，按 `(距离, row_id)` 排序取 TopK。
- **带过滤的检索**：除常规探测外，额外把「满足过滤条件的候选行所属的全部聚类」纳入探测范围，
  避免过滤把候选压缩到少数聚类后被常规探测漏掉。
- **索引维护**：表数据发生变化时索引标记为脏，下次 ANN 查询前重建；
  `CREATE VECTOR INDEX` 显式建立索引；进程重启时从持久化记录重建，重建完成后才输出 ready。

### 召回率

召回率定义为 `|返回集合 ∩ 真值集合| / min(K, N)`，真值在「满足过滤条件的集合」内按
`(L2 距离, 主键)` 计算。实测 10,000 条 × 128 维、1,000 条 TopK 查询：

| 指标 | 结果 |
| --- | --- |
| Recall@10（无过滤） | 1.00 |
| Recall@10（有等值过滤） | 1.00 |
| 阈值 | 0.95 |

## 4. 事务处理

- **自动提交**：未显式 `BEGIN` 时，每条成功的修改语句立即落盘（`fsync` 列文件、WAL、向量文件）
  并向 `commit.log` 追加提交点。
- **显式事务**：`BEGIN` 后记录各表的 checkpoint（文件长度、下一 row_id、内存映射备份）。
  事务内的查询与向量检索读取当前内存状态，因此能看到本事务已完成的修改。
  `COMMIT` 时统一落盘并追加提交点；`ROLLBACK` 时把文件截断回 checkpoint 并重放 WAL 恢复内存状态。
- **语句原子性**：一条语句先整体校验再落盘。例如多行 `INSERT` 中任一行失败
  （主键重复、维度不符等），整条语句不产生任何部分修改；显式事务中此前成功的语句仍可继续提交或回滚。

## 5. 提交持久化与恢复流程

`commit.log` 每行记录一次提交时各表的状态：WAL 长度、向量文件长度、各列文件长度。

启动流程：

1. 读取 `catalog.json` 得到表结构；
2. 读取 `commit.log` 得到每个表最后一次提交的长度标记；
3. 把各文件截断到该标记——**提交点之后的数据一律丢弃**，因此进程被强杀时，
   未提交事务不会部分生效，已确认提交的数据不丢；
4. 重放 WAL 重建存活记录集合；
5. 重建近似索引（从持久化记录重建，不依赖索引文件本身正确）；
6. 全部完成后才输出 `{"type":"ready"}`。

崩溃模型限定为进程被强制终止，不模拟断电、磁盘损坏或文件被外部篡改。

## 6. 协议实现要点

- 启动先完成恢复与索引重建，再输出 `{"type":"ready"}` 并 flush，之后才读第一条请求。
- 每条请求恰好一个响应，原样回传 `id`（保持其 JSON 类型，数字不转字符串）。
- stdout 只有协议报文；日志一律写 stderr。响应后立即 flush。
- `shutdown` 回滚未提交事务、落盘、返回 `{"id":N,"ok":true}`，随后以退出码 0 结束，
  关闭响应后不再有任何 stdout 输出。
- 预期内的 SQL 错误返回 `ok:false` 与非空 `error.code`，进程不退出；任何未预期异常
  也被兜底为 `internal_error` 响应，不会终止进程。

## 7. 性能自测记录

本机（Apple Silicon，未安装 Docker，本地 `--command` 模式）实测：

| 项目 | 规模 | 耗时 |
| --- | --- | --- |
| 批量写入 | 10,000 条 × 128 维，每批 100 条并提交 | 2.46 s |
| 索引建立 | 同上数据 | 0.04 s |
| TopK 查询 | 1,000 条，K = 10（一半带等值过滤） | 2.83 s |

### 优化过程中的两个关键问题

**其一：逐行 JSON 解析导致查询耗时 187 s。** 最初每行记录以 JSON 整体存储并逐行解析，
每条查询要解析 10,000 条记录。改为列式定长槽位后，过滤比较与距离计算都可用 numpy
向量化完成，查询耗时降到 2.83 s。

**其二：memmap 读到过期页。** 写入高 `row_id` 槽位时需要先扩展文件，Python 缓冲文件对象
在此场景下写入位置与预期不一致（实测写 row_id=2 后该槽位仍为 0，而磁盘数据正确）。
改为全程 `os.pwrite` / `os.pread` 后问题消失。

## 8. 已实现功能

- 表结构：`CREATE TABLE`（`BIGINT`、`VARCHAR(n)`、`VECTOR(n)`，单主键，单向量列）。
- 数据操作：`INSERT`（含多行）、`SELECT`（含 `SELECT *`）、`UPDATE`、`DELETE`。
- 条件查询：`=`、`<`、`<=`、`>`、`>=`、`AND`、`ORDER BY`（`ASC`/`DESC`）、`LIMIT`。
- 向量检索：`L2_DISTANCE`、精确 TopK、带等值过滤的 TopK、`CREATE VECTOR INDEX`、
  `exact` / `ann` 双模式。
- 持久化：正常关闭与重启后内容一致；崩溃恢复保留已提交数据。
- 事务：`BEGIN` / `COMMIT` / `ROLLBACK`，事务内可见自身修改。

## 9. 已知限制

- 不支持 `NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、外键、嵌套事务、主键更新
  （按题面要求返回 `unsupported_sql`）。
- `ORDER BY` 只支持单列或多列的普通列与距离表达式；未实现 `NULLS FIRST/LAST`。
- 字符串排序按 Unicode 码点顺序，不做语言相关排序（符合题面要求）。
- 向量索引为 IVF 平坦索引，未做乘积量化等压缩手段，索引文件体积与向量数据同量级。
- `VARCHAR` 字符串堆只增不回收，删除记录后其字节不被复用；长期大量删除会造成文件膨胀。
- 本地开发环境无 Docker，镜像未在本机构建验证；功能与协议均通过题包自带的
  `protocol_check` 在本地进程模式下验证，正式提测前需在有 Docker 的环境补一次镜像级自测。

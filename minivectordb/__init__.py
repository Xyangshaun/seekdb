"""MiniVectorDB：轻量向量数据库。

模块划分：
    errors.py       错误类型与错误码
    sqlparser.py    SQL 子集解析器
    storage.py      持久化存储（WAL + 崩溃恢复）
    vectorindex.py  自研 IVF 向量索引
    engine.py       SQL 执行引擎与事务
    stdio_server.py stdin/stdout JSONL 协议服务端
"""

__all__ = ["engine", "errors", "sqlparser", "storage", "stdio_server", "vectorindex"]

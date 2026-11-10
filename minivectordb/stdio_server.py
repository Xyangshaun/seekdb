"""stdin/stdout JSONL 协议服务端。

规则（见 STDIO_PROTOCOL.md）：
    启动先输出 {"type":"ready"} 并 flush，且必须在等待第一条请求之前。
    每个请求恰好一个响应，原样回传 id 与其 JSON 类型。
    stdout 只输出协议报文，日志一律写 stderr。
    响应后立即 flush；shutdown 响应后不再输出任何内容，退出码 0。
"""

import json
import sys

from .engine import Engine
from .errors import MiniVectorDBError, InvalidRequest


class Server:
    def __init__(self, data_dir=None, stdout=None, stderr=None):
        self.data_dir = data_dir or "/data"
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.engine = None

    def log(self, message):
        self.stderr.write(str(message) + "\n")
        self.stderr.flush()

    def write_message(self, payload):
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.stdout.write(text + "\n")
        self.stdout.flush()

    def start(self):
        # 恢复必须先完成，之后才能报告就绪
        self.engine = Engine(self.data_dir)
        self.engine.rebuild_indexes()
        self.log("minivectordb ready at %s" % self.data_dir)
        self.write_message({"type": "ready"})

    def handle(self, request):
        if not isinstance(request, dict):
            return {"id": None, "ok": False,
                    "error": {"code": "invalid_request", "message": "request must be an object"}}
        ident = request.get("id")
        op = request.get("op")
        try:
            if op == "shutdown":
                result = self.engine.execute("ROLLBACK") if self.engine.in_transaction else None
                self.engine.close()
                return {"id": ident, "ok": True}
            if op != "execute":
                raise InvalidRequest("unsupported op %r" % op)
            sql = request.get("sql")
            search_mode = request.get("search_mode") or "exact"
            if search_mode not in ("exact", "ann"):
                raise InvalidRequest("unsupported search_mode %r" % search_mode)
            outcome = self.engine.execute(sql, search_mode)
            response = {"id": ident, "ok": True}
            if "affected_rows" in outcome:
                response["affected_rows"] = outcome["affected_rows"]
            if "columns" in outcome:
                response["columns"] = outcome["columns"]
                response["rows"] = outcome["rows"]
            return response
        except MiniVectorDBError as exc:
            return {"id": ident, "ok": False, "error": exc.to_dict()}
        except Exception as exc:  # 兜底：任何异常都不得终止进程
            self.log("internal error: %r" % (exc,))
            return {"id": ident, "ok": False,
                    "error": {"code": "internal_error", "message": str(exc)}}

    def serve(self):
        self.start()
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            text = line.strip()
            if not text:
                continue
            try:
                request = json.loads(text)
            except ValueError as exc:
                self.log("invalid request json: %r" % (exc,))
                self.write_message({"id": None, "ok": False,
                                   "error": {"code": "invalid_request",
                                             "message": "malformed JSON request"}})
                continue
            response = self.handle(request)
            self.write_message(response)
            if isinstance(request, dict) and request.get("op") == "shutdown":
                break
        return 0


def main(argv=None):
    import os
    argv = list(argv if argv is not None else sys.argv[1:])
    data_dir = os.environ.get("VECTORDB_DATA_DIR") or "/data"
    for index, arg in enumerate(argv):
        if arg == "--data-dir" and index + 1 < len(argv):
            data_dir = argv[index + 1]
        elif arg.startswith("--data-dir="):
            data_dir = arg.split("=", 1)[1]
    os.makedirs(data_dir, exist_ok=True)
    server = Server(data_dir)
    return server.serve()

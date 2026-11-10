#!/usr/bin/env python3
"""崩溃恢复专项测试。

覆盖题面 3.2 的恢复要求：
    1. 强制终止进程后，已确认提交的数据保留；
    2. 未提交事务不部分生效（整体回滚）；
    3. 恢复途中再次终止，检查重复恢复的幂等性；
    4. 重启后向量检索结果与崩溃前一致。
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Client:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "minivectordb", "--data-dir", data_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=ROOT)
        self.seq = 0
        self.ready = json.loads(self.proc.stdout.readline().decode("utf-8"))

    def request(self, sql=None, mode="exact", op="execute"):
        self.seq += 1
        payload = {"id": self.seq, "op": op}
        if sql is not None:
            payload.update(sql=sql, search_mode=mode)
        self.proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        if op == "shutdown":
            return json.loads(self.proc.stdout.readline().decode("utf-8"))
        return json.loads(self.proc.stdout.readline().decode("utf-8"))

    def kill(self):
        """模拟崩溃：直接强杀进程，不发 shutdown。"""
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait(timeout=10)

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
        except Exception:
            pass


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("[%s] %s %s" % (status, label, detail))
    return condition


def main():
    data_dir = tempfile.mkdtemp(prefix="minivectordb-crash-")
    ok = True
    try:
        # --- 阶段 1：提交一批数据 ---
        client = Client(data_dir)
        ok &= check("握手 ready", client.ready == {"type": "ready"}, repr(client.ready))
        client.request("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), "
                       "content VARCHAR(16), embedding VECTOR(2));")
        client.request("INSERT INTO docs VALUES (1,'a','一','[0,0]'), (2,'a','二','[1,0]'), "
                       "(3,'b','三','[0,1]');")
        committed = client.request("SELECT id, content FROM docs ORDER BY id;")
        ok &= check("提交后可见 3 条", len(committed["rows"]) == 3, repr(committed["rows"]))

        # --- 阶段 2：开事务改数据，然后崩溃 ---
        client.request("BEGIN;")
        client.request("UPDATE docs SET content = '改' WHERE id = 1;")
        client.request("DELETE FROM docs WHERE id = 2;")
        client.request("INSERT INTO docs VALUES (4,'c','四','[2,2]');")
        before_kill = client.request("SELECT id, content FROM docs ORDER BY id;")
        # 事务内：id=1 改为「改」、id=2 被删除、新增 id=4，共 3 条可见
        ok &= check("事务内可见自身修改",
                    before_kill["rows"] == [[1, "改"], [3, "三"], [4, "四"]],
                    repr(before_kill["rows"]))
        client.kill()
        client.close()

        # --- 阶段 3：重启，未提交事务应整体回滚 ---
        client = Client(data_dir)
        after = client.request("SELECT id, content FROM docs ORDER BY id;")
        ok &= check("崩溃后未提交事务整体回滚",
                    after["rows"] == [[1, "一"], [2, "二"], [3, "三"]],
                    repr(after["rows"]))
        vec = client.request("SELECT id, L2_DISTANCE(embedding, '[0,0]') AS distance "
                             "FROM docs ORDER BY distance LIMIT 2;")
        # 回滚后向量为 id=1 [0,0] 距离 0；id=2 [1,0] 与 id=3 [0,1] 距离均为 1，
        # 按主键升序取 id=2
        ok &= check("崩溃后向量检索结果正确",
                    vec["rows"] == [[1, 0.0], [2, 1.0]], repr(vec["rows"]))

        # --- 阶段 4：再次崩溃（恢复途中重复终止）---
        client.request("INSERT INTO docs VALUES (5,'d','五','[3,3]');")
        client.kill()
        client.close()
        client = Client(data_dir)
        again = client.request("SELECT id FROM docs ORDER BY id;")
        ok &= check("二次崩溃后已提交数据仍在",
                    again["rows"] == [[1], [2], [3], [5]], repr(again["rows"]))

        # --- 阶段 5：再次崩溃（重复恢复幂等）---
        client.kill()
        client.close()
        client = Client(data_dir)
        third = client.request("SELECT id, content FROM docs ORDER BY id;")
        ok &= check("重复恢复幂等",
                    third["rows"] == [[1, "一"], [2, "二"], [3, "三"], [5, "五"]],
                    repr(third["rows"]))

        # --- 阶段 6：正常关闭后重启 ---
        client.request("INSERT INTO docs VALUES (6,'e','六','[4,4]');")
        client.request(op="shutdown")
        client.close()
        client = Client(data_dir)
        final = client.request("SELECT id FROM docs ORDER BY id;")
        ok &= check("正常关闭重启后数据完整",
                    final["rows"] == [[1], [2], [3], [5], [6]], repr(final["rows"]))
        client.request(op="shutdown")
        client.close()
    finally:
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)

    print("\n结果：%s" % ("全部通过" if ok else "存在失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

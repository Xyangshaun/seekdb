"""MiniVectorDB 的 SQL 子集解析器。

只覆盖赛题约定的语法：
CREATE TABLE / CREATE VECTOR INDEX / INSERT / SELECT / UPDATE / DELETE /
BEGIN / COMMIT / ROLLBACK。

不支持的语法一律抛 UnsupportedSQL，由上层转成 error.code=unsupported_sql。
"""

from .errors import UnsupportedSQL

KEYWORDS = {
    "CREATE", "TABLE", "VECTOR", "INDEX", "ON", "INSERT", "INTO", "VALUES",
    "SELECT", "FROM", "WHERE", "ORDER", "BY", "ASC", "DESC", "LIMIT", "UPDATE",
    "SET", "DELETE", "BEGIN", "COMMIT", "ROLLBACK", "START", "TRANSACTION",
    "WORK", "PRIMARY", "KEY", "AND", "AS", "NOT", "NULL", "IS", "LIKE",
}

PUNCT = ["(", ")", ",", ";", "<=", ">=", "!=", "=", "<", ">", "*", "+", "-", "/"]


class Token:
    __slots__ = ("kind", "value", "pos")

    def __init__(self, kind, value, pos):
        self.kind = kind
        self.value = value
        self.pos = pos

    def __repr__(self):
        return "Token(%s,%r)" % (self.kind, self.value)


def tokenize(sql):
    tokens = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if sql.startswith("--", i) or ch == "#":
            j = sql.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j < 0:
                raise UnsupportedSQL("unterminated block comment")
            i = j + 2
            continue
        if ch == "'" or ch == '"':
            quote = ch
            j = i + 1
            buf = []
            while True:
                if j >= n:
                    raise UnsupportedSQL("unterminated string literal")
                c = sql[j]
                if c == "\\" and j + 1 < n:
                    nxt = sql[j + 1]
                    buf.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
                    j += 2
                    continue
                if c == quote:
                    if j + 1 < n and sql[j + 1] == quote:
                        buf.append(quote)
                        j += 2
                        continue
                    j += 1
                    break
                buf.append(c)
                j += 1
            tokens.append(Token("str", "".join(buf), i))
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and sql[i + 1].isdigit()):
            j = i
            seen_dot = False
            seen_exp = False
            while j < n:
                c = sql[j]
                if c.isdigit():
                    j += 1
                elif c == "." and not seen_dot and not seen_exp:
                    seen_dot = True
                    j += 1
                elif c in "eE" and not seen_exp and j + 1 < n and (
                        sql[j + 1].isdigit() or (sql[j + 1] in "+-" and j + 2 < n and sql[j + 2].isdigit())):
                    seen_exp = True
                    j += 2
                else:
                    break
            raw = sql[i:j]
            if seen_dot or seen_exp:
                tokens.append(Token("num", float(raw), i))
            else:
                tokens.append(Token("num", int(raw), i))
            i = j
            continue
        if ch.isalpha() or ch == "_" or ord(ch) > 127:
            j = i
            while j < n and (sql[j].isalnum() or sql[j] in "_$" or ord(sql[j]) > 127):
                j += 1
            word = sql[i:j]
            upper = word.upper()
            tokens.append(Token("kw" if upper in KEYWORDS else "ident", word, i))
            i = j
            continue
        matched = None
        for p in PUNCT:
            if sql.startswith(p, i):
                matched = p
                break
        if matched is None:
            raise UnsupportedSQL("unexpected character %r" % ch)
        tokens.append(Token("punct", matched, i))
        i += len(matched)
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0

    def peek(self, offset=0):
        pos = self.index + offset
        return self.tokens[pos] if pos < len(self.tokens) else None

    def next(self):
        token = self.peek()
        if token is None:
            raise UnsupportedSQL("unexpected end of statement")
        self.index += 1
        return token

    def at_kw(self, *words):
        token = self.peek()
        return token is not None and token.kind == "kw" and token.value.upper() in words

    def at_punct(self, *chars):
        token = self.peek()
        return token is not None and token.kind == "punct" and token.value in chars

    def expect_kw(self, *words):
        token = self.next()
        if token.kind != "kw" or token.value.upper() not in words:
            raise UnsupportedSQL("expected %s, got %r" % ("/".join(words), token.value))
        return token

    def expect_punct(self, char):
        token = self.next()
        if token.kind != "punct" or token.value != char:
            raise UnsupportedSQL("expected %r, got %r" % (char, token.value))
        return token

    def expect_ident(self):
        token = self.next()
        if token.kind != "ident":
            raise UnsupportedSQL("expected identifier, got %r" % token.value)
        return token.value

    def end_of_statement(self):
        token = self.peek()
        if token is None:
            return True
        if token.kind == "punct" and token.value == ";":
            self.index += 1
            return True
        return False

    # ---- 各语句 ----

    def parse_statement(self):
        token = self.peek()
        if token is None:
            raise UnsupportedSQL("empty statement")
        if token.kind != "kw":
            raise UnsupportedSQL("unsupported statement: %r" % token.value)
        word = token.value.upper()
        if word == "CREATE":
            self.index += 1
            if self.at_kw("TABLE"):
                self.index += 1
                return self.parse_create_table()
            if self.at_kw("VECTOR"):
                self.index += 1
                self.expect_kw("INDEX")
                return self.parse_create_vector_index()
            raise UnsupportedSQL("unsupported CREATE variant")
        if word == "INSERT":
            self.index += 1
            return self.parse_insert()
        if word == "SELECT":
            self.index += 1
            return self.parse_select()
        if word == "UPDATE":
            self.index += 1
            return self.parse_update()
        if word == "DELETE":
            self.index += 1
            return self.parse_delete()
        if word in ("BEGIN", "START"):
            self.index += 1
            if self.at_kw("TRANSACTION"):
                self.index += 1
            elif self.at_kw("WORK"):
                self.index += 1
            self._end_statement()
            return {"type": "begin"}
        if word == "COMMIT":
            self.index += 1
            if self.at_kw("WORK"):
                self.index += 1
            self._end_statement()
            return {"type": "commit"}
        if word == "ROLLBACK":
            self.index += 1
            if self.at_kw("WORK"):
                self.index += 1
            self._end_statement()
            return {"type": "rollback"}
        raise UnsupportedSQL("unsupported statement: %s" % token.value)

    def _end_statement(self):
        """消费语句结束符；后面还有内容说明是第二条语句，必须报错。"""
        if not self.end_of_statement():
            raise UnsupportedSQL("trailing tokens after statement")

    def parse_create_table(self):
        name = self.expect_ident()
        self.expect_punct("(")
        columns = []
        primary = None
        while True:
            if self.at_kw("PRIMARY"):
                self.index += 1
                self.expect_kw("KEY")
                self.expect_punct("(")
                primary = self.expect_ident()
                self.expect_punct(")")
            else:
                col_name = self.expect_ident()
                col_type = self.parse_column_type()
                is_pk = False
                if self.at_kw("PRIMARY"):
                    self.index += 1
                    self.expect_kw("KEY")
                    is_pk = True
                columns.append({"name": col_name, "type": col_type["type"],
                                "length": col_type["length"], "primary_key": is_pk})
                if is_pk and primary is None:
                    primary = col_name
            if self.at_punct(","):
                self.index += 1
                continue
            break
        self.expect_punct(")")
        if primary is None:
            for col in columns:
                if col["primary_key"]:
                    primary = col["name"]
        if not self.end_of_statement():
            raise UnsupportedSQL("trailing tokens after CREATE TABLE")
        return {"type": "create_table", "name": name, "columns": columns, "primary_key": primary}

    def parse_column_type(self):
        token = self.next()
        if token.kind not in ("kw", "ident"):
            raise UnsupportedSQL("expected column type, got %r" % token.value)
        word = token.value.upper()
        if word == "BIGINT":
            return {"type": "bigint", "length": None}
        if word in ("VARCHAR", "VECTOR"):
            length = None
            if self.at_punct("("):
                self.index += 1
                num = self.next()
                if num.kind != "num" or not isinstance(num.value, int):
                    raise UnsupportedSQL("expected integer length")
                length = num.value
                self.expect_punct(")")
            return {"type": word.lower(), "length": length}
        raise UnsupportedSQL("unsupported column type: %s" % token.value)

    def parse_create_vector_index(self):
        name = self.expect_ident()
        self.expect_kw("ON")
        table = self.expect_ident()
        self.expect_punct("(")
        column = self.expect_ident()
        self.expect_punct(")")
        if not self.end_of_statement():
            raise UnsupportedSQL("trailing tokens after CREATE VECTOR INDEX")
        return {"type": "create_vector_index", "name": name, "table": table, "column": column}

    def parse_insert(self):
        self.expect_kw("INTO")
        table = self.expect_ident()
        columns = None
        if self.at_punct("("):
            self.index += 1
            columns = []
            while True:
                columns.append(self.expect_ident())
                if self.at_punct(","):
                    self.index += 1
                    continue
                break
            self.expect_punct(")")
        self.expect_kw("VALUES")
        rows = []
        while True:
            self.expect_punct("(")
            values = []
            while True:
                values.append(self.parse_literal())
                if self.at_punct(","):
                    self.index += 1
                    continue
                break
            self.expect_punct(")")
            rows.append(values)
            if self.at_punct(","):
                self.index += 1
                continue
            break
        if not self.end_of_statement():
            raise UnsupportedSQL("trailing tokens after INSERT")
        return {"type": "insert", "table": table, "columns": columns, "rows": rows}

    def parse_literal(self):
        token = self.next()
        if token.kind == "num":
            return {"kind": "number", "value": token.value}
        if token.kind == "str":
            return {"kind": "string", "value": token.value}
        if token.kind == "punct" and token.value == "-":
            inner = self.parse_literal()
            if inner["kind"] != "number":
                raise UnsupportedSQL("cannot negate non-numeric literal")
            return {"kind": "number", "value": -inner["value"]}
        if token.kind == "punct" and token.value == "[":
            parts = []
            while not self.at_punct("]"):
                parts.append(self.parse_literal())
                if self.at_punct(","):
                    self.index += 1
            self.expect_punct("]")
            values = []
            for part in parts:
                if part["kind"] != "number":
                    raise UnsupportedSQL("vector literal must contain numbers")
                values.append(part["value"])
            return {"kind": "vector", "value": values}
        if token.kind == "kw" and token.value.upper() == "NULL":
            raise UnsupportedSQL("NULL is not supported")
        raise UnsupportedSQL("unsupported literal: %r" % token.value)

    def parse_select(self):
        projections = []
        while True:
            if self.at_punct("*"):
                self.index += 1
                projections.append({"expr": {"kind": "star"}, "alias": None})
            else:
                expr = self.parse_expr()
                alias = None
                if self.at_kw("AS"):
                    self.index += 1
                    alias = self.expect_ident()
                elif self.peek() is not None and self.peek().kind == "ident":
                    alias = self.expect_ident()
                projections.append({"expr": expr, "alias": alias})
            if self.at_punct(","):
                self.index += 1
                continue
            break
        self.expect_kw("FROM")
        table = self.expect_ident()
        where = None
        if self.at_kw("WHERE"):
            self.index += 1
            where = self.parse_condition()
        order_by = None
        if self.at_kw("ORDER"):
            self.index += 1
            self.expect_kw("BY")
            order_by = []
            while True:
                expr = self.parse_expr()
                descending = False
                if self.at_kw("DESC"):
                    self.index += 1
                    descending = True
                elif self.at_kw("ASC"):
                    self.index += 1
                order_by.append({"expr": expr, "desc": descending})
                if self.at_punct(","):
                    self.index += 1
                    continue
                break
        # ORDER BY 可以引用投影别名（如 AS distance），
        # 在此把别名与投影表达式绑定，供执行层解析。
        aliases = {}
        for projection in projections:
            alias = projection.get("alias")
            if alias:
                aliases[alias.lower()] = projection["expr"]
        if aliases:
            for term in order_by or []:
                expr = term["expr"]
                if expr["kind"] == "column":
                    target = aliases.get(expr["name"].lower())
                    if target is not None:
                        term["expr"] = target
        limit = None
        if self.at_kw("LIMIT"):
            self.index += 1
            token = self.next()
            if token.kind != "num" or not isinstance(token.value, int):
                raise UnsupportedSQL("LIMIT expects an integer")
            limit = token.value
        if not self.end_of_statement():
            raise UnsupportedSQL("trailing tokens after SELECT")
        return {"type": "select", "table": table, "projections": projections,
                "where": where, "order_by": order_by, "limit": limit}

    def parse_expr(self):
        token = self.peek()
        if token is None:
            raise UnsupportedSQL("unexpected end of expression")
        if token.kind == "ident" and token.value.upper() == "L2_DISTANCE":
            self.index += 1
            self.expect_punct("(")
            column = self.expect_ident()
            self.expect_punct(",")
            literal = self.parse_literal()
            self.expect_punct(")")
            if literal["kind"] != "vector" and literal["kind"] != "string":
                raise UnsupportedSQL("L2_DISTANCE requires a vector literal")
            return {"kind": "l2_distance", "column": column, "query": literal}
        if token.kind == "ident":
            self.index += 1
            return {"kind": "column", "name": token.value}
        if token.kind == "num" or token.kind == "str":
            self.index += 1
            return {"kind": "constant", "value": token.value}
        raise UnsupportedSQL("unsupported expression: %r" % token.value)

    def parse_condition(self):
        conditions = []
        while True:
            left = self.parse_operand()
            token = self.next()
            if token.kind != "punct" or token.value not in ("=", "<", "<=", ">", ">="):
                raise UnsupportedSQL("unsupported comparison operator: %r" % token.value)
            right = self.parse_operand()
            conditions.append({"left": left, "op": token.value, "right": right})
            if self.at_kw("AND"):
                self.index += 1
                continue
            break
        return {"type": "and", "conditions": conditions}

    def parse_operand(self):
        token = self.peek()
        if token is None:
            raise UnsupportedSQL("unexpected end of condition")
        if token.kind == "ident" and token.value.upper() == "L2_DISTANCE":
            return self.parse_expr()
        if token.kind == "ident":
            self.index += 1
            return {"kind": "column", "name": token.value}
        if token.kind == "num" or token.kind == "str":
            self.index += 1
            return {"kind": "constant", "value": token.value}
        if token.kind == "punct" and token.value == "-":
            self.index += 1
            inner = self.parse_operand()
            if inner["kind"] != "constant" or not isinstance(inner["value"], (int, float)):
                raise UnsupportedSQL("cannot negate non-numeric operand")
            return {"kind": "constant", "value": -inner["value"]}
        raise UnsupportedSQL("unsupported operand: %r" % token.value)

    def parse_update(self):
        table = self.expect_ident()
        self.expect_kw("SET")
        assignments = []
        while True:
            column = self.expect_ident()
            self.expect_punct("=")
            assignments.append({"column": column, "value": self.parse_literal()})
            if self.at_punct(","):
                self.index += 1
                continue
            break
        where = None
        if self.at_kw("WHERE"):
            self.index += 1
            where = self.parse_condition()
        if not self.end_of_statement():
            raise UnsupportedSQL("trailing tokens after UPDATE")
        return {"type": "update", "table": table, "assignments": assignments, "where": where}

    def parse_delete(self):
        self.expect_kw("FROM")
        table = self.expect_ident()
        where = None
        if self.at_kw("WHERE"):
            self.index += 1
            where = self.parse_condition()
        if not self.end_of_statement():
            raise UnsupportedSQL("trailing tokens after DELETE")
        return {"type": "delete", "table": table, "where": where}


def parse(sql):
    if sql is None:
        raise UnsupportedSQL("missing SQL")
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise UnsupportedSQL("empty statement")
    parser = Parser(tokenize(text))
    statement = parser.parse_statement()
    if not parser.end_of_statement():
        raise UnsupportedSQL("trailing tokens after statement")
    return statement

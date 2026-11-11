"""SQL 执行引擎：表管理、CRUD、事务、向量化查询。

事务模型：
    autocommit 模式下每条修改语句立即提交（落盘 + 写 commit.log）。
    显式 BEGIN 后进入事务，修改只写入日志不落盘；COMMIT 时一次性落盘并写提交点；
    ROLLBACK 时按 checkpoint 截断文件并重放 WAL 回滚内存状态。
    崩溃（进程被强杀）时，commit.log 最后一个提交点之后的数据被丢弃，
    因此「已确认提交不丢、未提交不部分生效」。

查询性能：
    普通列过滤与向量距离都用 numpy 向量化完成，避免逐行解释。
"""

import json

import numpy as np

from . import sqlparser
from .errors import (DimensionMismatch, DuplicateKey, InvalidVector, InvalidValue,
                     NoSuchColumn, NoSuchTable, TableExists, TypeMismatch,
                     UnsupportedSQL, ValueTooLong)
from .storage import Database
from .vectorindex import VectorIndex


class Transaction:
    def __init__(self):
        self.checkpoints = {}
        self.live_backup = {}
        self.row_pk_backup = {}
        self.next_row_backup = {}


class Engine:
    def __init__(self, data_dir):
        self.db = Database(data_dir)
        self.db.open()
        self.in_transaction = False
        self.transaction = None
        self.indexes = {}

    # ---- 启动后重建索引 ----

    def rebuild_indexes(self):
        """恢复时重建近似索引；重建完成前不得报告就绪。"""
        for name, table in self.db.tables.items():
            if not table.vector_column:
                continue
            index = VectorIndex(table.vector_dim)
            if not index.load(table.index_dir) or not index.row_ids:
                index = self._build_index(table)
            self.indexes[name] = index

    def _build_index(self, table):
        index = VectorIndex(table.vector_dim)
        row_ids = np.array(sorted(table.live.values()), dtype=np.int64)
        if len(row_ids) == 0:
            index.build(np.zeros((0, table.vector_dim), dtype=np.float32), [])
            return index
        vectors = table.vector_block(row_ids)
        index.build(vectors, row_ids.tolist())
        index.save(table.index_dir)
        return index

    # ---- 事务 ----

    def begin(self):
        if self.in_transaction:
            raise UnsupportedSQL("nested transactions are not supported")
        self.in_transaction = True
        self.transaction = Transaction()
        for name, table in self.db.tables.items():
            self.transaction.checkpoints[name] = table.checkpoint()
            self.transaction.live_backup[name] = dict(table.live)
            self.transaction.row_pk_backup[name] = dict(table.row_pk)
            self.transaction.next_row_backup[name] = table.next_row_id

    def commit(self):
        if not self.in_transaction:
            return
        self.db.commit()
        self.in_transaction = False
        self.transaction = None

    def rollback(self):
        if not self.in_transaction:
            return
        for name, checkpoint in self.transaction.checkpoints.items():
            table = self.db.tables.get(name)
            if table is None:
                continue
            table.restore(checkpoint)
            table.live = self.transaction.live_backup.get(name, {})
            table.row_pk = self.transaction.row_pk_backup.get(name, {})
            table.next_row_id = self.transaction.next_row_backup.get(name, 0)
        self.in_transaction = False
        self.transaction = None

    def _maybe_commit(self):
        if not self.in_transaction:
            self.db.commit()

    def _snapshot(self, table):
        if not self.in_transaction:
            return
        if table.name in self.transaction.checkpoints:
            return
        self.transaction.checkpoints[table.name] = table.checkpoint()
        self.transaction.live_backup[table.name] = dict(table.live)
        self.transaction.row_pk_backup[table.name] = dict(table.row_pk)
        self.transaction.next_row_backup[table.name] = table.next_row_id

    # ---- 分发 ----

    def execute(self, sql, search_mode="exact"):
        statement = sqlparser.parse(sql)
        kind = statement["type"]
        if kind == "create_table":
            return self.exec_create_table(statement)
        if kind == "create_vector_index":
            return self.exec_create_vector_index(statement)
        if kind == "insert":
            return self.exec_insert(statement)
        if kind == "select":
            return self.exec_select(statement, search_mode)
        if kind == "update":
            return self.exec_update(statement)
        if kind == "delete":
            return self.exec_delete(statement)
        if kind == "begin":
            self.begin()
            return {}
        if kind == "commit":
            self.commit()
            return {}
        if kind == "rollback":
            self.rollback()
            return {}
        raise UnsupportedSQL("unsupported statement type: %s" % kind)

    def _get_table(self, name):
        table = self.db.tables.get(name)
        if table is None:
            raise NoSuchTable("table %s does not exist" % name)
        return table

    # ---- DDL ----

    def exec_create_table(self, statement):
        name = statement["name"]
        if name in self.db.tables:
            raise TableExists("table %s already exists" % name)
        columns = statement["columns"]
        primary = statement["primary_key"]
        if primary is None:
            raise UnsupportedSQL("a BIGINT primary key is required")
        vector_columns = [c for c in columns if c["type"] == "vector"]
        if len(vector_columns) > 1:
            raise UnsupportedSQL("at most one vector column per table")
        schema = {
            "columns": [{"name": c["name"], "type": c["type"], "length": c["length"]}
                        for c in columns],
            "primary_key": primary,
            "vector_column": vector_columns[0]["name"] if vector_columns else None,
            "vector_dim": vector_columns[0]["length"] if vector_columns else 0,
        }
        self.db.create_table(name, schema)
        self.db.commit()
        if vector_columns:
            self.indexes[name] = self._build_index(self.db.tables[name])
        return {}

    def exec_create_vector_index(self, statement):
        table = self._get_table(statement["table"])
        column = statement["column"]
        if table.vector_column != column:
            raise NoSuchColumn("table %s has no vector column %s" % (table.name, column))
        self.indexes[table.name] = self._build_index(table)
        self.db.commit()
        return {}

    # ---- INSERT ----

    def exec_insert(self, statement):
        table = self._get_table(statement["table"])
        columns = statement["columns"]
        if columns is None:
            columns = [c["name"] for c in table.schema["columns"]]
        column_types = {c["name"]: c for c in table.schema["columns"]}
        for name in columns:
            if name not in column_types:
                raise NoSuchColumn("unknown column %s" % name)
        if len(set(columns)) != len(columns):
            raise InvalidValue("duplicate column in INSERT")
        self._snapshot(table)
        # 先整体校验，任何一行失败都不能留下部分修改
        prepared = []
        for row in statement["rows"]:
            if len(row) != len(columns):
                raise InvalidValue("column count does not match value count")
            values = {}
            for name, literal in zip(columns, row):
                values[name] = self._coerce(table, column_types[name], literal)
            pk = values.get(table.primary_key)
            if pk is None:
                raise InvalidValue("primary key must not be NULL")
            if pk in table.live:
                raise DuplicateKey("duplicate primary key %r" % pk)
            prepared.append((pk, values))
        affected = 0
        for pk, values in prepared:
            row_id = table.allocate_row_id()
            vector = values.get(table.vector_column) if table.vector_column else None
            if vector is not None:
                table.write_vector(row_id, vector)
            table.write_row(row_id, pk, values)
            affected += 1
        self._invalidate_index(table)
        self._maybe_commit()
        return {"affected_rows": affected}

    def _coerce(self, table, column, literal):
        kind = literal["kind"]
        ctype = column["type"]
        if ctype == "bigint":
            if kind != "number":
                raise TypeMismatch("column %s expects BIGINT" % column["name"])
            value = literal["value"]
            if isinstance(value, float):
                if not value.is_integer():
                    raise InvalidValue("BIGINT expects an integer")
                value = int(value)
            return value
        if ctype == "varchar":
            if kind == "string":
                value = literal["value"]
            elif kind == "number":
                value = str(literal["value"]) if isinstance(literal["value"], int) \
                    else repr(literal["value"])
            else:
                raise TypeMismatch("column %s expects VARCHAR" % column["name"])
            limit = column.get("length")
            if limit is not None and len(value) > limit:
                raise ValueTooLong("value exceeds VARCHAR(%d)" % limit)
            return value
        if ctype == "vector":
            if kind == "vector":
                values = literal["value"]
            elif kind == "string":
                values = self._parse_vector_text(literal["value"])
            else:
                raise InvalidVector("column %s expects VECTOR" % column["name"])
            dim = table.schema.get("vector_dim") or 0
            if dim and len(values) != dim:
                raise DimensionMismatch("expected %d dimensions, got %d" % (dim, len(values)))
            try:
                array = np.asarray(values, dtype=np.float32)
            except (TypeError, ValueError, OverflowError):
                raise InvalidVector("vector contains values that cannot be represented as float32")
            if not np.all(np.isfinite(array)):
                raise InvalidVector("vector must contain finite float32 values")
            return [float(x) for x in array.tolist()]
        raise TypeMismatch("unsupported column type")

    def _parse_vector_text(self, text):
        try:
            value = json.loads(text)
        except ValueError:
            raise InvalidVector("invalid vector literal")
        if not isinstance(value, list):
            raise InvalidVector("invalid vector literal")
        return value

    # ---- UPDATE / DELETE ----

    def _row_values(self, table, row_id):
        """读取单行的全部普通列值（向量列单独存放，不在此读取）。"""
        index = np.array([row_id], dtype=np.int64)
        values = {}
        for column in table.schema["columns"]:
            if column["type"] == "vector":
                continue
            values[column["name"]] = table.column_values(column["name"], index)[0]
        return values

    def exec_update(self, statement):
        table = self._get_table(statement["table"])
        column_types = {c["name"]: c for c in table.schema["columns"]}
        for assignment in statement["assignments"]:
            if assignment["column"] not in column_types:
                raise NoSuchColumn("unknown column %s" % assignment["column"])
        self._snapshot(table)
        targets = self._match_rows(table, statement["where"])
        # 先整体校验：任何一行失败都不能留下部分修改，改主键的冲突也要先查清
        prepared = []
        for pk, row_id in targets:
            values = self._row_values(table, row_id)
            updated = dict(values)
            for assignment in statement["assignments"]:
                updated[assignment["column"]] = self._coerce(
                    table, column_types[assignment["column"]], assignment["value"])
            new_pk = updated[table.primary_key]
            if new_pk in table.live and table.live[new_pk] != row_id:
                raise DuplicateKey("duplicate primary key %r" % new_pk)
            prepared.append((pk, row_id, updated))
        affected = 0
        for pk, row_id, updated in prepared:
            new_pk = updated[table.primary_key]
            new_row_id = table.allocate_row_id()
            vector = updated.get(table.vector_column) if table.vector_column else None
            if vector is None and table.vector_column:
                # 未修改向量列时沿用原向量
                vector = table.read_vector(row_id)
            if vector is not None:
                table.write_vector(new_row_id, np.asarray(vector, dtype=np.float32).tolist())
            table.write_row(new_row_id, new_pk, updated)
            if new_pk != pk:
                # 旧主键必须显式写删除记录，崩溃恢复时才不会让旧行复活
                table.delete_row(row_id, pk)
            else:
                table.row_pk.pop(row_id, None)
            affected += 1
        self._invalidate_index(table)
        self._maybe_commit()
        return {"affected_rows": affected}

    def exec_delete(self, statement):
        table = self._get_table(statement["table"])
        self._snapshot(table)
        targets = self._match_rows(table, statement["where"])
        for pk, row_id in targets:
            table.delete_row(row_id, pk)
        self._invalidate_index(table)
        self._maybe_commit()
        return {"affected_rows": len(targets)}

    # ---- SELECT ----

    def exec_select(self, statement, search_mode):
        table = self._get_table(statement["table"])
        resolved = [self._resolve_projection(table, p) for p in statement["projections"]]
        order_terms = [self._resolve_order_term(table, t) for t in statement["order_by"] or []]
        for term in order_terms:
            if term["kind"] == "distance":
                term["query_array"] = self._coerce_vector_for_query(table, term["query"])
        limit = statement["limit"]
        has_distance = any(r["kind"] == "distance" for r in resolved)
        if has_distance:
            return self._select_vector(table, statement, resolved, limit,
                                      search_mode, order_terms)
        return self._select_plain(table, statement, resolved, order_terms, limit)

    def _resolve_projection(self, table, projection):
        expr = projection["expr"]
        names = {c["name"] for c in table.schema["columns"]}
        if expr["kind"] == "star":
            return {"kind": "star", "name": "*", "alias": None}
        if expr["kind"] == "column":
            if expr["name"] not in names:
                raise NoSuchColumn("unknown column %s" % expr["name"])
            return {"kind": "column", "name": expr["name"],
                    "alias": projection.get("alias") or expr["name"]}
        if expr["kind"] == "l2_distance":
            if table.vector_column != expr["column"]:
                raise NoSuchColumn("table %s has no vector column %s"
                                   % (table.name, expr["column"]))
            return {"kind": "distance", "name": expr["column"],
                    "query": expr["query"]["value"],
                    "alias": projection.get("alias") or "distance"}
        if expr["kind"] == "constant":
            return {"kind": "constant", "value": expr["value"],
                    "alias": projection.get("alias") or "constant"}
        raise UnsupportedSQL("unsupported projection")

    def _resolve_order_term(self, table, term):
        expr = term["expr"]
        names = {c["name"] for c in table.schema["columns"]}
        if expr["kind"] == "column":
            if expr["name"] not in names:
                raise NoSuchColumn("unknown column %s" % expr["name"])
            return {"kind": "column", "name": expr["name"], "desc": term["desc"]}
        if expr["kind"] == "l2_distance":
            if table.vector_column != expr["column"]:
                raise NoSuchColumn("table %s has no vector column %s"
                                   % (table.name, expr["column"]))
            return {"kind": "distance", "query": expr["query"]["value"], "desc": term["desc"]}
        raise UnsupportedSQL("unsupported ORDER BY expression")

    def _select_plain(self, table, statement, resolved, order_terms, limit):
        matches = self._match_rows(table, statement["where"])
        columns = self._output_columns(table, resolved)
        if not order_terms:
            # 无 ORDER BY：按主键升序输出（集合比较时顺序不影响结果）
            matches.sort(key=lambda item: item[0])
            if limit is not None:
                matches = matches[:limit]
            return {"columns": columns, "rows": self._project_rows(table, resolved, matches)}
        # ORDER BY：取出相关列后向量化排序
        row_ids = np.array([row_id for _, row_id in matches], dtype=np.int64)
        if len(row_ids) == 0:
            return {"columns": columns, "rows": []}
        keys = []
        for term in order_terms:
            if term["kind"] == "column":
                column = table.stores[term["name"]]
                values = column.compare_array(row_ids)
                keys.append((term, values))
            else:
                query_vec = term.get("query_array")
                if query_vec is None:
                    query_vec = self._coerce_vector_for_query(table, term["query"])
                vectors = table.vector_block(row_ids)
                diff = vectors - query_vec
                distances = np.sqrt((diff * diff).sum(axis=1)).astype(np.float64)
                keys.append((term, distances))
        order = self._sort_indices(keys, row_ids, table)
        if limit is not None:
            order = order[:limit]
        picked = [(int(table.row_pk.get(row_ids[i], 0)), int(row_ids[i]))
                  for i in order]
        return {"columns": columns, "rows": self._project_rows(table, resolved, picked)}

    def _sort_indices(self, keys, row_ids, table):
        """按多列排序，等距时按主键升序。返回下标数组。"""
        import functools

        def comparator(i, j):
            for term, values in keys:
                if values is None:
                    continue
                a = values[i]
                b = values[j]
                if isinstance(a, bytes) or isinstance(b, bytes):
                    if a == b:
                        continue
                    result = -1 if a < b else 1
                else:
                    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                        a, b = int(a), int(b)
                    else:
                        a, b = float(a), float(b)
                    if a == b:
                        continue
                    result = -1 if a < b else 1
                if term["desc"]:
                    result = -result
                return result
            pk_a = int(table.row_pk.get(row_ids[i], 0))
            pk_b = int(table.row_pk.get(row_ids[j], 0))
            return -1 if pk_a < pk_b else (0 if pk_a == pk_b else 1)

        return sorted(range(len(row_ids)), key=functools.cmp_to_key(comparator))

    def _project_rows(self, table, resolved, matches):
        if not matches:
            return []
        row_ids = np.array([row_id for _, row_id in matches], dtype=np.int64)
        columns = {}
        for item in resolved:
            if item["kind"] == "column":
                columns[item["name"]] = table.column_values(item["name"], row_ids)
        out = []
        for position in range(len(matches)):
            row = []
            for item in resolved:
                if item["kind"] == "column":
                    row.append(columns[item["name"]][position])
                elif item["kind"] == "constant":
                    row.append(item["value"])
                else:
                    row.append(None)
            out.append(row)
        return out

    def _output_columns(self, table, resolved):
        columns = []
        for item in resolved:
            if item["kind"] == "star":
                for column in table.schema["columns"]:
                    columns.append(column["name"])
            else:
                columns.append(item["alias"])
        return columns

    # ---- 向量查询 ----

    def _select_vector(self, table, statement, resolved, limit, search_mode, order_terms):
        distance_term = next(r for r in resolved if r["kind"] == "distance")
        query_vector = self._coerce_vector_for_query(table, distance_term["query"])
        matches = self._match_rows(table, statement["where"])
        columns = self._output_columns(table, resolved)
        if not matches:
            return {"columns": columns, "rows": []}
        k = limit if limit is not None else len(matches)
        row_ids = np.array([row_id for _, row_id in matches], dtype=np.int64)
        if search_mode == "ann":
            hits = self._ann_search(table, query_vector, k, row_ids)
        else:
            hits = self._exact_search(table, query_vector, k, row_ids)
        if limit is not None:
            hits = hits[:limit]
        rows = []
        for row_id, distance in hits:
            row = []
            for item in resolved:
                if item["kind"] == "distance":
                    row.append(distance)
                elif item["kind"] == "column":
                    row.append(table.column_values(item["name"], np.array([row_id], dtype=np.int64))[0])
                elif item["kind"] == "constant":
                    row.append(item["value"])
                else:
                    row.append(None)
            rows.append(row)
        return {"columns": columns, "rows": rows}

    def _coerce_vector_for_query(self, table, query):
        dim = table.schema.get("vector_dim") or 0
        if isinstance(query, str):
            query = self._parse_vector_text(query)
        if dim and len(query) != dim:
            raise DimensionMismatch("expected %d dimensions, got %d" % (dim, len(query)))
        try:
            array = np.asarray(query, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            raise InvalidVector("query vector is not representable as float32")
        if not np.all(np.isfinite(array)):
            raise InvalidVector("query vector must be finite")
        return array

    def _exact_search(self, table, query_vector, k, row_ids):
        vectors = table.vector_block(row_ids)
        diff = vectors - query_vector
        distances = np.sqrt((diff * diff).sum(axis=1)).astype(np.float64)
        pk_values = table.column_values(table.primary_key, row_ids)
        order = np.lexsort((np.asarray(pk_values, dtype=np.int64), distances))[:k]
        return [(int(row_ids[i]), float(distances[i])) for i in order]

    def _ann_search(self, table, query_vector, k, row_ids):
        index = self.indexes.get(table.name)
        live_row_ids = set(table.live.values())
        if index is None or index.dirty or set(index.row_ids) != live_row_ids:
            index = self._build_index(table)
            self.indexes[table.name] = index
        if index.vectors is None or index.vectors.shape[0] == 0:
            return []
        allowed = set(row_ids.tolist())
        candidates = index.search(query_vector, k, allowed=allowed)
        return [(row_id, distance) for row_id, distance in candidates]

    def _invalidate_index(self, table):
        index = self.indexes.get(table.name)
        if index is not None:
            index.dirty = True

    # ---- WHERE 匹配（向量化） ----

    def _match_rows(self, table, where):
        """返回满足条件的 [(pk, row_id)] 列表。"""
        row_ids = table.live_row_ids()
        if len(row_ids) == 0:
            return []
        if where is None:
            pks = table.column_values(table.primary_key, row_ids)
            return list(zip(pks, row_ids.tolist()))
        mask = np.ones(len(row_ids), dtype=bool)
        for condition in where["conditions"]:
            left, right, op = (condition["left"], condition["right"], condition["op"])
            mask &= self._compare_mask(table, row_ids, left, op, right)
        selected = row_ids[mask]
        pks = table.column_values(table.primary_key, selected)
        return list(zip(pks, selected.tolist()))

    def _compare_mask(self, table, row_ids, left, op, right):
        if left["kind"] == "column":
            column = table.stores.get(left["name"])
            if column is None:
                raise NoSuchColumn("unknown column %s" % left["name"])
            values = column.compare_array(row_ids)
            constant = self._operand_constant(table, right)
            if constant is None:
                # 右侧是列：逐行比较（少见场景）
                other = table.stores[right["name"]].compare_array(row_ids)
                mask = np.zeros(len(values), dtype=bool)
                for i in range(len(values)):
                    mask[i] = self._compare_scalar(values[i], op, other[i])
                return mask
            return self._compare_vector(values, op, constant)
        constant = self._operand_constant(table, left)
        if constant is None:
            raise UnsupportedSQL("unsupported comparison")
        other = table.stores[right["name"]].compare_array(row_ids)
        return self._compare_vector(other, op, constant)

    def _operand_constant(self, table, operand):
        if operand["kind"] == "constant":
            return operand["value"]
        return None

    def _compare_vector(self, values, op, constant):
        """values 为 numpy 数组或 bytes 列表，与常量比较。"""
        if isinstance(values, list):
            encoded = constant.encode("utf-8") if isinstance(constant, str) else constant
            mask = np.zeros(len(values), dtype=bool)
            for i, item in enumerate(values):
                mask[i] = self._compare_scalar(item, op, encoded)
            return mask
        array = np.asarray(values)
        if isinstance(constant, str):
            raise TypeMismatch("cannot compare BIGINT column with a string")
        if op == "=":
            return array == constant
        if op == "<":
            return array < constant
        if op == "<=":
            return array <= constant
        if op == ">":
            return array > constant
        if op == ">=":
            return array >= constant
        raise UnsupportedSQL("unsupported operator %s" % op)

    def _compare_scalar(self, left, op, right):
        if left is None or right is None:
            return False
        if isinstance(left, bytes) != isinstance(right, bytes):
            if isinstance(left, bytes) or isinstance(right, bytes):
                return False
        try:
            if op == "=":
                return left == right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
        except TypeError:
            return False
        raise UnsupportedSQL("unsupported operator %s" % op)

    # ---- 关闭 ----

    def close(self):
        self.rollback()
        for table in self.db.tables.values():
            table.flush()
        self.db.close()

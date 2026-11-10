"""错误类型与错误码。

错误码命名与公开题包 mysqltest/cases/*.result 中的 error 字段一致：
duplicate_key / dimension_mismatch / unsupported_sql。
"""


class MiniVectorDBError(Exception):
    code = "internal_error"

    def __init__(self, message, code=None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code

    def to_dict(self):
        return {"code": self.code, "message": self.message}


class UnsupportedSQL(MiniVectorDBError):
    code = "unsupported_sql"


class DuplicateKey(MiniVectorDBError):
    code = "duplicate_key"


class DimensionMismatch(MiniVectorDBError):
    code = "dimension_mismatch"


class NoSuchTable(MiniVectorDBError):
    code = "no_such_table"


class NoSuchColumn(MiniVectorDBError):
    code = "no_such_column"


class TableExists(MiniVectorDBError):
    code = "table_exists"


class ValueTooLong(MiniVectorDBError):
    code = "value_too_long"


class InvalidVector(MiniVectorDBError):
    code = "invalid_vector"


class InvalidValue(MiniVectorDBError):
    code = "invalid_value"


class TypeMismatch(MiniVectorDBError):
    code = "type_mismatch"


class InvalidRequest(MiniVectorDBError):
    code = "invalid_request"

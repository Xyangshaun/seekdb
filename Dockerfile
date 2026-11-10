# MiniVectorDB 评测容器
#
# 构建：
#   docker build -t minivectordb:latest .
#
# 运行（前台、保持标准输入打开、与判题器通过 stdin/stdout 通信）：
#   docker run -i --network=none --mount type=bind,src=/local/data,dst=/data minivectordb:latest
#
# 不使用 -d，不分配 -t，不暴露或映射端口。

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VECTORDB_DATA_DIR=/data

# 依赖在构建阶段安装完成，运行时无需联网。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-numpy \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 源码：完整数据库实现，随仓库提交。
COPY minivectordb/ /app/minivectordb/

# 运行说明与设计记录。
COPY README.md /app/README.md
COPY docs/ /app/docs/

RUN mkdir -p /data

# 默认以前台模式启动 stdin/stdout 服务，数据目录固定为 /data。
ENTRYPOINT ["python3", "-m", "minivectordb"]
CMD ["--data-dir", "/data"]

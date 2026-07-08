# FROM 预构建的基础镜像(见 harbor/README.md 的 docker build 步骤)。
FROM architectureiq-harbor:base
RUN mkdir -p /workspace && \
    python -m architectureiq bbl-init --run-dir /workspace/game --family optimizer --hidden sgd
WORKDIR /workspace

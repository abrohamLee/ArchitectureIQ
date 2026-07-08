# FROM 预构建的基础镜像(见 harbor/README.md 的 docker build 步骤)。
FROM architectureiq-harbor:base
RUN mkdir -p /workspace && \
    python -m architectureiq dx-init --run-dir /workspace/game --pathology vanishing_grad
WORKDIR /workspace

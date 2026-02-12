# ============================================================================
# MinerU Server - Enterprise AI Data Preprocessing (CUDA 13.0 Edition)
# Optimized for Aliyun Mirrors with Network Resilience
# Updated: PaddleOCR >= 3.4.0, PaddleX >= 3.4.1
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image (CUDA 13.0 + Ubuntu 24.04)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS base

# 1. [关键配置] 切换阿里云源 + 强力抗网络波动配置
#    增加 Retries(重试 20次) 和 Timeout(5分钟)，专门解决 403/Connection Refused 问题
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    # 写入 APT 优化配置
    echo 'Acquire::http::Pipeline-Depth "0";' > /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::http::No-Cache "true";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::BrokenProxy "true";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::Retries "20";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::http::Timeout "300";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::https::Timeout "300";' >> /etc/apt/apt.conf.d/99custom

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PIP_CACHE_DIR=/root/.cache/pip

# 2. 安装系统依赖 (合并层以减小体积，利用 Retries 机制防止下载中断)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip python3.12-venv \
    git wget curl \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1 libgstreamer1.0-0 ffmpeg antiword pandoc \
    libreoffice libreoffice-writer libreoffice-calc libreoffice-impress \
    fonts-noto-cjk fonts-wqy-zenhei fonts-wqy-microhei fontconfig \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# 3. Python 环境配置 (软链接)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# [升级 pip] 使用阿里云源
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade --ignore-installed pip "setuptools>=75.0.0" wheel \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --break-system-packages

# -----------------------------------------------------------------------------
# Stage 2: Dependencies Installation (严格顺序控制)
# -----------------------------------------------------------------------------
FROM base AS dependencies
WORKDIR /tmp
COPY backend/requirements.txt /tmp/requirements.txt

# 统一使用阿里云 Pypi 源
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/

# Step 1: [地基] 锁定 NumPy (1.26.4)
# 必须最先安装，防止后续包将其升级到 2.0 导致 Paddle 崩溃
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing Foundation (NumPy 1.26.4) ===" && \
    pip install "numpy==1.26.4" "opencv-python-headless>=4.10.0.84" \
    -i ${PIP_MIRROR} \
    --break-system-packages

# Step 2: [框架A] 安装 PyTorch (2.10.0+cu130)
# 必须在 FlashAttention 之前安装
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing PyTorch 2.10.0+cu130 ===" && \
    pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130 \
    --break-system-packages

# Step 3: [插件] 安装 FlashAttention (Custom Wheel)
# 依赖已存在的 Torch 2.10 环境
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing FlashAttention ===" && \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl \
    --break-system-packages

# Step 4: [框架B] 安装 PaddlePaddle (3.3.0+cu130)
# 放在 Torch 之后，确保它不会拉取冲突的 protobuf/numpy
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing PaddlePaddle 3.3.0 ===" && \
    pip install paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
    --break-system-packages

# Step 5: [业务库] 安装 MinerU Core & Transformers & PaddleX
# 关键: 使用 --no-deps 防止它们覆盖我们精心配置的 Torch/NumPy
# 更新: PaddleOCR >= 3.4.0, PaddleX >= 3.4.1
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing MinerU & PaddleX (No Deps) ===" && \
    pip install "mineru[core]>=2.7.6" \
                "albumentations>=1.4.11" \
                "albucore>=0.0.15" \
                "transformers==4.57.6" \
                "tokenizers>=0.22.0" \
                "doclayout-yolo>=0.0.2" \
                "paddleocr>=3.4.0" \
                "paddlex>=3.4.1" \
    --no-deps \
    -i ${PIP_MIRROR} \
    --break-system-packages

# Step 6: [补全] 安装 requirements.txt 中的剩余依赖
# 这一步会补全 MinerU/PaddleX 需要的小型依赖 (requests, click, visualdl等)
# 我们先用 sed 再次清理一遍，以防万一文件中还残留了 torch/paddle/paddlex
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing Remaining Requirements ===" && \
    sed -i '/torch/d; /paddle/d; /nvidia/d; /opencv/d; /numpy/d; /albumentations/d; /transformers/d; /tokenizers/d; /vllm/d; /flash_attn/d; /paddlex/d' /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt \
    -i ${PIP_MIRROR} \
    --break-system-packages

# 验证环境 (仅验证 Import，不验证 CUDA 设备，因为构建时可能无 GPU)
RUN python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    python -c "import paddle; print(f'✅ Paddle: {paddle.__version__}')" && \
    python -c "import paddleocr; print(f'✅ PaddleOCR: {paddleocr.__version__}')" && \
    python -c "import paddlex; print(f'✅ PaddleX: {paddlex.__version__}')"

# -----------------------------------------------------------------------------
# Stage 3: Application Runtime
# -----------------------------------------------------------------------------
FROM dependencies AS application

WORKDIR /app
COPY backend/ /app/backend/
COPY pyproject.toml /app/
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY scripts/init-models.sh /usr/local/bin/init-models.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh /usr/local/bin/init-models.sh && \
    mkdir -p /app/models /app/data/uploads /app/data/output /app/logs

WORKDIR /app/backend

# 暴露端口
# 8000: API Server
# 8001: LitServe Worker
# 8002: MCP Server
EXPOSE 8000 8001 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["api", "python", "api_server.py"]

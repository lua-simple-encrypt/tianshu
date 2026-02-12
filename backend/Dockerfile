# ============================================================================
# MinerU Server - Enterprise AI Data Preprocessing (CUDA 13.0 Edition)
# Optimized for Aliyun Mirrors with Network Resilience
# Fix: huggingface-hub conflict & libcuda.so.1 import error during build
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image (CUDA 13.0 + Ubuntu 24.04)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS base

# 1. [关键配置] 切换阿里云源 + 强力抗网络波动配置
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    # 写入 APT 优化配置，增加重试次数和超时时间
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

# 2. 安装系统依赖
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
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing Foundation (NumPy 1.26.4) ===" && \
    pip install "numpy==1.26.4" "opencv-python-headless>=4.11.0.86" \
    -i ${PIP_MIRROR} \
    --break-system-packages

# Step 2: [框架A] 安装 PyTorch (2.10.0+cu130)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing PyTorch 2.10.0+cu130 ===" && \
    pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130 \
    --break-system-packages

# Step 3: [插件] 安装 FlashAttention
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing FlashAttention ===" && \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl \
    --break-system-packages

# Step 4: [框架B] 安装 PaddlePaddle (3.3.0+cu130)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing PaddlePaddle 3.3.0 ===" && \
    pip install paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
    --break-system-packages

# Step 5: [业务库] 安装 MinerU, PaddleX, PaddleOCR[all]
# [重要] 增加了 "huggingface-hub<1.0" 锁定以解决与 transformers 的冲突
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing MinerU & Paddle Ecosystem (No Deps) ===" && \
    pip install "huggingface-hub<1.0" \
                "mineru[core]>=2.7.6" \
                "albumentations>=1.4.11" \
                "albucore>=0.0.15" \
                "transformers==4.57.6" \
                "tokenizers>=0.22.0" \
                "doclayout-yolo==0.0.4" \
                "paddleocr[all]>=3.4.0" \
                "paddlex>=3.4.1" \
                "pdftext>=0.6.3" \
                "json-repair>=0.46.2" \
                "magika>=0.6.2" \
                "scikit-image>=0.25.0" \
                "fast-langdetect>=0.2.3" \
    --no-deps \
    -i ${PIP_MIRROR} \
    --break-system-packages

# Step 6: [补全] 安装 requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing Remaining Requirements ===" && \
    sed -i '/torch/d; /paddle/d; /nvidia/d; /opencv/d; /numpy/d; /albumentations/d; /transformers/d; /tokenizers/d; /vllm/d; /flash_attn/d; /paddlex/d; /doclayout/d; /scikit-image/d; /paddleocr/d' /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt \
    -i ${PIP_MIRROR} \
    --break-system-packages

# Step 7: [最终验证] 验证环境 (修正：移除会引起报错的 import paddle)
# 构建阶段没有 libcuda.so.1 驱动，Paddle 无法完成 import。
RUN python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" && \
    echo "Verifying Paddle installation via metadata..." && \
    pip show paddlepaddle-gpu | grep Version && \
    pip show paddleocr | grep Version && \
    pip show paddlex | grep Version && \
    echo "✅ Paddle packages verified (Import check skipped during build stage)"

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

EXPOSE 8000 8001 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["api", "python", "api_server.py"]

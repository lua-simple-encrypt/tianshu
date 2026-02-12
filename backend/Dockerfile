# ============================================================================
# MinerU Server - Enterprise AI Data Preprocessing (CUDA 13.0 Edition)
# Optimized for: MinerU 2.7.6 (1.2B VLM Support)
# Updated: Fix boto3, pypdf missing & transformers conflict
# ============================================================================

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS base

# 1. [网络优化] 阿里云源 + 抗干扰配置
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    echo 'Acquire::Retries "20"; Acquire::http::Timeout "300";' > /etc/apt/apt.conf.d/99custom

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PIP_CACHE_DIR=/root/.cache/pip

# 2. [系统修复] 安装底层库
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip python3.12-venv \
    git wget curl ffmpeg antiword pandoc poppler-utils \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1 libgl1-mesa-glx libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. [系统修复] 安装中文字体 (解决渲染文本丢失)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-core fonts-noto-cjk fonts-noto-cjk-extra \
    fonts-wqy-zenhei fonts-wqy-microhei fontconfig \
    libreoffice \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# 4. Python 环境配置
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade --ignore-installed pip "setuptools>=75.0.0" wheel \
    -i https://mirrors.aliyun.com/pypi/simple/ --break-system-packages

# -----------------------------------------------------------------------------
# Stage 2: Dependencies Installation
# -----------------------------------------------------------------------------
FROM base AS dependencies
WORKDIR /tmp
COPY backend/requirements.txt /tmp/requirements.txt
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/

# Step 1: [地基] 锁定 NumPy 和 OpenCV
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy==1.26.4" "opencv-python-headless>=4.11.0.86" -i ${PIP_MIRROR} --break-system-packages

# Step 2: [框架A] PyTorch 2.10.0+cu130
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

# Step 3: [扩展] FlashAttention (匹配 Torch 2.10)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl --break-system-packages

# Step 4: [框架B] PaddlePaddle 3.3.0+cu130
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --break-system-packages

# Step 5: [业务库] 核心组件安装 (解决 boto3, pypdf, huggingface-hub 冲突)
# 这里强制补全了 MinerU 2.7.6 所有的“软依赖”
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "huggingface-hub<1.0" \
                "mineru[core]>=2.7.6" \
                "paddleocr[all]>=3.4.0" \
                "paddlex>=3.4.1" \
                "transformers==4.57.6" \
                "tokenizers>=0.22.0" \
                "doclayout-yolo==0.0.4" \
                "pdftext>=0.6.3" \
                "boto3>=1.34.0" \
                "pypdf>=5.1.0" \
                "json-repair>=0.46.2" \
                "magika>=0.6.2" \
                "fast-langdetect>=0.2.3" \
                "scikit-image>=0.25.0" \
                "shapely>=2.0.0" \
                "pyclipper>=1.3.0" \
    --no-deps -i ${PIP_MIRROR} --break-system-packages

# Step 6: [补全] 安装 requirements.txt (过滤掉已手动安装的重型包)
RUN --mount=type=cache,target=/root/.cache/pip \
    sed -i '/torch/d; /paddle/d; /nvidia/d; /opencv/d; /numpy/d; /albumentations/d; /transformers/d; /tokenizers/d; /vllm/d; /flash_attn/d; /paddlex/d; /doclayout/d; /paddleocr/d; /huggingface/d; /boto3/d; /pypdf/d' /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt -i ${PIP_MIRROR} --break-system-packages

# Step 7: [最终验证] 检查关键包安装情况 (跳过构建时不可用的 paddle import)
RUN python -c "import torch; import pypdf; import boto3; print('✅ Core Deps Verified')" && \
    pip show paddlepaddle-gpu paddleocr paddlex mineru | grep Version

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

# 暴露端口 (API, Worker, MCP)
EXPOSE 8000 8001 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["api", "python", "api_server.py"]

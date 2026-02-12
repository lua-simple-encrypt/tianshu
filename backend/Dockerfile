# ============================================================================
# MinerU Server - Enterprise AI Data Preprocessing (CUDA 13.0 Edition)
# Optimized for: MinerU 2.5-2509-1.2B (VLM Edition)
# Fixes: libGL.so.1, Missing CJK Fonts, Build Network Resilience
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

# 2. [关键修复] 安装系统依赖
# 包含: libgl1-mesa-glx (解决 libGL.so 错误), antiword/pandoc (MinerU 必需)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip python3.12-venv \
    git wget curl ffmpeg antiword pandoc poppler-utils \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1 libgl1-mesa-glx libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. [关键修复] 安装中文字体 (解决 MinerU >=2.0 渲染文本丢失问题)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-core \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    fontconfig \
    libreoffice \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# 4. Python 环境配置
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# 升级 pip 并使用阿里云源
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

# Step 1: 地基
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy==1.26.4" "opencv-python-headless>=4.11.0.86" -i ${PIP_MIRROR} --break-system-packages

# Step 2: PyTorch 2.10
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

# Step 3: FlashAttention
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl --break-system-packages

# Step 4: PaddlePaddle 3.3
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --break-system-packages

# Step 5: 业务库 (适配 1.2B VLM)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "huggingface-hub<1.0" \
                "mineru[core]>=2.7.6" \
                "paddleocr[all]>=3.4.0" \
                "paddlex>=3.4.1" \
                "transformers==4.57.6" \
                "doclayout-yolo==0.0.4" \
                "pdftext>=0.6.3" \
    --no-deps -i ${PIP_MIRROR} --break-system-packages

# Step 6: 补全
RUN --mount=type=cache,target=/root/.cache/pip \
    sed -i '/torch/d; /paddle/d; /nvidia/d; /opencv/d; /numpy/d; /albumentations/d; /transformers/d; /tokenizers/d; /vllm/d; /flash_attn/d; /paddlex/d; /doclayout/d; /paddleocr/d' /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt -i ${PIP_MIRROR} --break-system-packages

# Step 7: 最终验证 (绕过构建时 libcuda.so 限制)
RUN python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    pip show paddlepaddle-gpu paddleocr | grep Version && \
    echo "✅ Environment Ready for MinerU 1.2B"

# -----------------------------------------------------------------------------
# Stage 3: Application Runtime
# -----------------------------------------------------------------------------
FROM dependencies AS application
WORKDIR /app
COPY backend/ /app/backend/
COPY pyproject.toml /app/
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && mkdir -p /app/models /app/data/uploads /app/data/output /app/logs
WORKDIR /app/backend
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["api", "python", "api_server.py"]

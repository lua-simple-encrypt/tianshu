#!/bin/bash
# ============================================================================
# MinerU Tianshu Backend Installer (CUDA 13.0 Edition)
# Updated: PaddleOCR[all] (Full Features)
# ============================================================================

set -e

echo "📦 Installing MinerU Tianshu Backend Dependencies..."
echo "🚀 Target Environment: CUDA 13.0 | Torch 2.10 | Paddle 3.3 | PaddleOCR[all]"
echo ""

# ... (Step 1 - Step 6 保持不变，请直接使用之前发的内容) ...

TORCH_URL="https://download.pytorch.org/whl/cu130"
PADDLE_URL="https://www.paddlepaddle.org.cn/packages/stable/cu130/"
FLASH_ATTN_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl"
PIP_MIRROR="https://mirrors.aliyun.com/pypi/simple/"

# (此处省略 Step 1 - Step 6，与之前脚本完全一致)

# ============================================================================
# Step 7: MinerU & Core Apps (No Dependencies)
# ============================================================================
echo ""
echo "[Step 7/8] Installing MinerU, PaddleX & Transformers..."
echo "  > Using --no-deps to protect Torch/NumPy versions..."

# Updated list to include paddleocr[all]
pip install "mineru[core]>=2.7.6" \
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
    -i $PIP_MIRROR

echo "✓ Core applications installed"

# ============================================================================
# Step 8: Remaining Dependencies
# ============================================================================
echo ""
echo "[Step 8/8] Resolving remaining dependencies..."
echo "  > Cleaning requirements.txt to avoid conflicts..."

cd "$(dirname "$0")" || exit

sed '/torch/d; /paddle/d; /nvidia/d; /opencv/d; /numpy/d; /albumentations/d; /transformers/d; /tokenizers/d; /vllm/d; /flash_attn/d; /paddlex/d; /doclayout/d; /scikit-image/d; /paddleocr/d' requirements.txt > requirements.tmp.txt

echo "  > Installing safe dependencies..."
pip install -r requirements.tmp.txt \
    -i $PIP_MIRROR \
    --default-timeout=600

rm requirements.tmp.txt
echo "✓ All dependencies resolved"

# ... (Step 9 验证部分保持不变) ...

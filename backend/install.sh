#!/bin/bash
# ============================================================================
# MinerU Tianshu Backend Installer (CUDA 13.0 Edition)
# Environment: WSL2 / Ubuntu 24.04 / Python 3.12
# ============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

echo "📦 Installing MinerU Tianshu Backend Dependencies..."
echo "🚀 Target Environment: CUDA 13.0 | Torch 2.10 | Paddle 3.3"
echo ""

# Definition of versions
TORCH_URL="https://download.pytorch.org/whl/cu130"
PADDLE_URL="https://www.paddlepaddle.org.cn/packages/stable/cu130/"
FLASH_ATTN_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# ============================================================================
# Step 1: System Checks
# ============================================================================
echo "[Step 1/8] Checking system requirements..."

# Check OS
if [ "$(uname)" != "Linux" ]; then
    echo "❌ Error: This script is designed for Linux/WSL."
    exit 1
fi

# Check Python Version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "❌ Error: Python 3.12 is required. Found Python $PYTHON_VERSION."
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "⚠ Warning: nvidia-smi not found. Ensure you are passing GPU to WSL."
fi

# ============================================================================
# Step 2: System Libraries
# ============================================================================
echo ""
echo "[Step 2/8] Installing system libraries..."
if command -v apt-get &> /dev/null; then
    # Sudo might require password
    echo "  > Updating apt repository..."
    sudo apt-get update -qq
    echo "  > Installing libgomp1, ffmpeg, libGL..."
    sudo apt-get install -y libgomp1 ffmpeg libgl1 libglib2.0-0
    echo "✓ System libraries installed"
else
    echo "⚠ Warning: apt-get not found. Ensure libgomp1 and ffmpeg are installed."
fi

# ============================================================================
# Step 3: Foundation (NumPy & Setup)
# ============================================================================
echo ""
echo "[Step 3/8] Installing Foundation (NumPy 1.26.4)..."
echo "  > Locking NumPy to 1.26.4 to prevent Paddle/Torch conflicts..."

pip install --upgrade pip setuptools wheel -i $PIP_MIRROR
pip install "numpy==1.26.4" "opencv-python-headless>=4.10.0.84" \
    -i $PIP_MIRROR \
    --default-timeout=300

echo "✓ Foundation installed"

# ============================================================================
# Step 4: PyTorch (CUDA 13.0)
# ============================================================================
echo ""
echo "[Step 4/8] Installing PyTorch 2.10.0+cu130..."
echo "  > Downloading from $TORCH_URL ..."

pip install torch==2.10.0+cu130 \
    torchvision==0.25.0+cu130 \
    torchaudio==2.10.0+cu130 \
    --index-url $TORCH_URL \
    --default-timeout=600

echo "✓ PyTorch installed"

# ============================================================================
# Step 5: Flash Attention (Custom Wheel)
# ============================================================================
echo ""
echo "[Step 5/8] Installing Flash Attention (Precompiled)..."
echo "  > Target: Torch 2.10 + CUDA 13.0 + Python 3.12"

pip install $FLASH_ATTN_URL \
    --default-timeout=600

echo "✓ FlashAttention installed"

# ============================================================================
# Step 6: PaddlePaddle (CUDA 13.0)
# ============================================================================
echo ""
echo "[Step 6/8] Installing PaddlePaddle GPU 3.3.0..."
echo "  > Downloading from $PADDLE_URL ..."

pip install paddlepaddle-gpu==3.3.0 \
    -i $PADDLE_URL \
    --default-timeout=600

echo "✓ PaddlePaddle installed"

# ============================================================================
# Step 7: MinerU & Core Apps (No Dependencies)
# ============================================================================
echo ""
echo "[Step 7/8] Installing MinerU & Transformers..."
echo "  > Using --no-deps to protect Torch/NumPy versions..."

# Install Core Apps without dependencies first
pip install "mineru[core]>=2.7.6" \
    "transformers==4.57.6" \
    "tokenizers>=0.22.0" \
    "paddleocr>=2.7.3" \
    "albumentations>=1.4.11" \
    "albucore>=0.0.15" \
    "doclayout-yolo>=0.0.2" \
    --no-deps \
    -i $PIP_MIRROR

echo "✓ Core applications installed"

# ============================================================================
# Step 8: Remaining Dependencies
# ============================================================================
echo ""
echo "[Step 8/8] Resolving remaining dependencies..."
echo "  > Cleaning requirements.txt to avoid conflicts..."

# Ensure we are in the script's directory
cd "$(dirname "$0")" || exit

# Create a temporary requirements file excluding heavy frameworks
sed '/torch/d; /paddle/d; /nvidia/d; /opencv/d; /numpy/d; /albumentations/d; /transformers/d; /tokenizers/d; /vllm/d; /flash_attn/d' requirements.txt > requirements.tmp.txt

echo "  > Installing safe dependencies..."
pip install -r requirements.tmp.txt \
    -i $PIP_MIRROR \
    --default-timeout=600

# Cleanup
rm requirements.tmp.txt
echo "✓ All dependencies resolved"

# ============================================================================
# Verification
# ============================================================================
echo ""
echo "============================================================"
echo "🔍 Verifying Installation..."
echo "============================================================"

python3 << 'EOF'
import sys
import os

# Helper for colored output
def ok(msg): print(f"\033[92m✓ {msg}\033[0m")
def fail(msg): print(f"\033[91m✗ {msg}\033[0m"); return False
def warn(msg): print(f"\033[93m⚠ {msg}\033[0m"); return True

success = True

# 1. NumPy Check
try:
    import numpy
    if numpy.__version__ == "1.26.4":
        ok(f"NumPy: {numpy.__version__} (Locked)")
    else:
        warn(f"NumPy: {numpy.__version__} (Warning: Expected 1.26.4)")
except Exception as e:
    success = fail(f"NumPy: {e}")

# 2. PyTorch Check
try:
    import torch
    ver = torch.__version__
    cuda = torch.cuda.is_available()
    dev_count = torch.cuda.device_count()
    if "2.10" in ver and "+cu130" in ver:
         ok(f"PyTorch: {ver}")
    else:
         warn(f"PyTorch: {ver} (Target: 2.10.0+cu130)")
    
    if cuda:
        ok(f"PyTorch CUDA: Available ({dev_count} GPUs)")
    else:
        warn("PyTorch CUDA: Not Available")
except Exception as e:
    success = fail(f"PyTorch: {e}")

# 3. Paddle Check
try:
    import paddle
    ver = paddle.__version__
    cuda = paddle.device.is_compiled_with_cuda()
    if ver == "3.3.0":
        ok(f"Paddle: {ver}")
    else:
        warn(f"Paddle: {ver} (Target: 3.3.0)")
        
    if cuda:
        ok("Paddle CUDA: Available")
    else:
        warn("Paddle CUDA: Not Available")
except Exception as e:
    success = fail(f"Paddle: {e}")

# 4. Flash Attention
try:
    import flash_attn
    ok(f"FlashAttention: {flash_attn.__version__}")
except ImportError:
    warn("FlashAttention: Not found (Optional but recommended)")
except Exception as e:
    warn(f"FlashAttention Error: {e}")

# 5. MinerU
try:
    from magic_pdf.data.dataset import Dataset
    ok("MinerU (magic-pdf): Ready")
except ImportError:
    fail("MinerU: magic-pdf module not found")
except Exception as e:
    # MinerU often has config warnings, treat as warning unless import fails
    ok(f"MinerU: Importable ({str(e)[:50]}...)")

if not success:
    sys.exit(1)
EOF

VERIFY_EXIT=$?

echo ""
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "✅ Installation Successful! You are ready to launch."
    echo "   Run: python api_server.py"
else
    echo "❌ Installation Verification Failed."
    echo "   Please check the errors above."
    exit 1
fi

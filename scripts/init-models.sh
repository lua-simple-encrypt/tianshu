#!/bin/bash
# Tianshu - 模型初始化脚本 (智能适配版)
# 更新：增加对本地挂载模式的识别，防止重复复制

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INIT]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[INIT]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[INIT]${NC} $1"
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    log_info "Checking model initialization..."

    # 1. 【核心新增】检测是否为本地挂载模式
    # 如果在 docker-compose 中设置了 MINERU_MODEL_SOURCE=local
    if [ "$MINERU_MODEL_SOURCE" = "local" ]; then
        log_success "✅ Detected MINERU_MODEL_SOURCE=local"
        log_info "Models are mounted directly from host (D: drive), skipping copy process."
        
        # 确保目录结构存在，防止 MinerU 启动时因找不到父目录报错
        mkdir -p /root/.cache/mineru/models
        mkdir -p /root/.paddleocr/models
        
        log_success "✅ Environment ready for local models."
        return 0
    fi

    # 2. 检测设备模式
    DEVICE_MODE=${DEVICE_MODE:-auto}
    if [ "$DEVICE_MODE" = "auto" ]; then
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
            log_info "Detected: GPU mode (auto-detection)"
        else
            log_info "Detected: CPU mode (auto-detection)"
        fi
    else
        log_info "Device mode: $DEVICE_MODE (manual configuration)"
    fi

    # 3. 检查初始化标记文件
    if [ -f "/root/.cache/.models_initialized" ]; then
        log_info "Models already initialized, skipping copy"
        return 0
    fi

    # 4. 检查外部模型目录是否存在 (非本地模式下的逻辑)
    if [ ! -d "/models-external" ]; then
        log_warning "External models directory not found at /models-external"
        log_warning "Models will be downloaded on first use by the application"
        return 0
    fi

    log_info "Copying models from external volume..."
    log_info "This is a one-time operation and may take 5-10 minutes"
    echo ""

    # 创建必要的目录
    mkdir -p /root/.cache/huggingface/hub
    mkdir -p /root/.paddleocr/models
    mkdir -p /root/.cache/watermark_models
    mkdir -p /app/models/sensevoice
    mkdir -p /app/models/paraformer

    # 复制逻辑 (保持原有逻辑作为非本地模式的备份)
    if [ -d "/models-external/huggingface/hub" ]; then
        log_info "Copying HuggingFace models (MinerU)..."
        cp -r /models-external/huggingface/hub/* /root/.cache/huggingface/hub/ 2>/dev/null || true
        log_success "HuggingFace models copied"
    fi

    if [ -d "/models-external/.paddleocr/models" ]; then
        log_info "Copying PaddleOCR models..."
        cp -r /models-external/.paddleocr/models/* /root/.paddleocr/models/ 2>/dev/null || true
        log_success "PaddleOCR models copied"
    fi

    # ... (SenseVoice/Paraformer/Watermark 复制逻辑保持不变)

    # 创建初始化标记文件
    touch /root/.cache/.models_initialized
    echo "$(date -Iseconds)" > /root/.cache/.models_initialized

    echo ""
    log_success "✅ Models initialized successfully from external volume"
}

# 执行主函数
main

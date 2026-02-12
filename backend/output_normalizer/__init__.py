"""
天枢输出规范化模块 (Production Ready)
功能：统一解析引擎输出，处理资产上传，生成 UI 友好型结果。

规范化标准：
1. 核心文档：result.md
2. 结构化数据：result.json
3. 图片资产：images/ (本地) 或 RustFS URL (云端)
4. 引用修正：自动纠正 Markdown 中的图片路径
"""

from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .base_output_normalizer import BaseOutputNormalizer
from .standard_output_normalizer import StandardOutputNormalizer
from .paddleocr_output_normalizer import PaddleOCROutputNormalizer

# 全局单例，避免重复初始化
_normalizers = {
    "standard": StandardOutputNormalizer(),
    "paddleocr-vl": PaddleOCROutputNormalizer()
}

def normalize_output(output_dir: Path, handle_method: str = "standard") -> Dict[str, Any]:
    """
    高级规范化入口函数

    逻辑：
    1. 如果 handle_method 为 "auto"，则根据目录结构智能判定。
    2. 执行物理文件更名、目录迁移。
    3. 如果开启了 RustFS，则触发图片上传并执行 Markdown 文本全局正则替换。

    Args:
        output_dir: 解析结果存放的物理路径
        handle_method: 指定处理器 ['standard', 'paddleocr-vl', 'auto']
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        logger.error(f"❌ Normalize failed: Directory not found {output_dir}")
        raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

    # --- 1. 智能格式判定 ---
    # PaddleOCR-VL 的典型特征是生成 page_1, page_2... 这种子目录
    is_paddle_pattern = any(output_dir.glob("page_*"))
    
    if handle_method == "auto" or handle_method == "standard":
        if is_paddle_pattern:
            logger.info("🤖 [Auto-Detect] Detected PaddleOCR-VL folder structure.")
            handle_method = "paddleocr-vl"
        else:
            handle_method = "standard"

    # --- 2. 选择规范化器 ---
    normalizer = _normalizers.get(handle_method, _normalizers["standard"])
    logger.info(f"🛠️  Normalizing output via [{handle_method}] strategy...")

    try:
        # 执行核心规范化逻辑（物理移动文件 -> 上传云端 -> 路径替换）
        result = normalizer.normalize(output_dir)
        
        logger.success(f"✅ Normalization complete for {output_dir.name}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Normalization process failed: {e}")
        # 如果规范化失败，返回原始路径至少保证任务不崩溃
        return {
            "result_path": str(output_dir),
            "status": "partial_success",
            "error": str(e)
        }

__all__ = [
    "BaseOutputNormalizer",
    "StandardOutputNormalizer",
    "PaddleOCROutputNormalizer",
    "normalize_output",
]

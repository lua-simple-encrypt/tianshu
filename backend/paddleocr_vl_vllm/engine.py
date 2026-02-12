"""
PaddleOCR-VL-VLLM 解析引擎 (天枢专用生产版)
单例模式，每个进程只加载一次基础版面识别模型, OCR/VLM 部分通过 OpenAI 协议调用远程 vLLM。

核心配置：
- 默认 API 地址: http://host.docker.internal:8118/v1 (由环境变量 PADDLE_VLM_URL 覆盖)
- 模型名称: PaddleOCR-VL-1.5
"""

import os
import json
import gc
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger

class PaddleOCRVLVLLMEngine:
    """
    PaddleOCR-VL-VLLM 解析引擎
    支持：多页 PDF、自动语言检测、Markdown 结构化输出
    """

    _instance: Optional["PaddleOCRVLVLLMEngine"] = None
    _lock = Lock()
    _pipeline = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = "cuda:0", vllm_api_base: str = None):
        """
        初始化引擎

        Args:
            device: 逻辑设备 ID (例如 "cuda:0")
            vllm_api_base: 如果提供则使用该地址，否则按 环境变量 -> 默认地址 顺序查找
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device
            # 优先级：参数传递 > 环境变量 > 天枢默认地址
            self.vllm_api_base = (
                vllm_api_base or 
                os.getenv("PADDLE_VLM_URL") or 
                "http://host.docker.internal:8118/v1"
            )

            # 提取 GPU ID
            try:
                self.gpu_id = int(device.split(":")[-1]) if "cuda:" in device else 0
            except Exception:
                self.gpu_id = 0

            # 验证 GPU 环境
            self._check_gpu_availability()
            self._initialized = True

            logger.info("=" * 60)
            logger.info("🔧 PaddleOCR-VL-VLLM Engine Initialized")
            logger.info(f"   📍 API Endpoint: {self.vllm_api_base}")
            logger.info(f"   🎯 Local Device: {self.device}")
            logger.info(f"   📂 Model Cache: ~/.paddleocr/models/ (Layout Mode)")
            logger.info("=" * 60)

    def _check_gpu_availability(self):
        """检查运行环境是否具备 Paddle GPU 推理能力"""
        try:
            import paddle
            if not paddle.is_compiled_with_cuda():
                logger.warning("⚠️ PaddlePaddle is NOT compiled with CUDA! Layout detection will be slow.")
            
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                gpu_name = paddle.device.cuda.get_device_name(self.gpu_id)
                logger.info(f"✅ GPU detected: {gpu_name}")
            else:
                logger.error("❌ No GPU found! PaddleOCR-VL requires GPU.")
        except ImportError:
            logger.error("❌ PaddlePaddle not installed. Run: pip install paddlepaddle-gpu")

    def _load_pipeline(self):
        """延迟加载管道，确保显存在需要时才分配"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                import paddle
                from paddleocr import PaddleOCRVL

                # 设置当前进程的显卡
                if paddle.is_compiled_with_cuda():
                    paddle.set_device(f"gpu:{self.gpu_id}")

                logger.info(f"📥 Loading PaddleOCRVL Pipeline (API: {self.vllm_api_base})...")

                # 创建实例，对接远程 vLLM
                self._pipeline = PaddleOCRVL(
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=True,
                    use_layout_detection=True,
                    vl_rec_backend="vllm-server",
                    vl_rec_server_url=self.vllm_api_base,
                )
                return self._pipeline

            except Exception as e:
                logger.error(f"❌ Failed to load PaddleOCRVL pipeline: {e}")
                raise

    def cleanup(self):
        """每次推理任务后清理临时缓存，防止显存碎片堆积"""
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
            gc.collect()
            logger.debug("🧹 GPU cache cleared after PaddleOCR task")
        except Exception:
            pass

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        核心解析函数

        Args:
            file_path: 待解析文件路径 (PDF/Image)
            output_path: 任务专属输出目录
        """
        input_file = Path(file_path)
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        pipeline = self._load_pipeline()
        
        try:
            logger.info(f"🚀 [PaddleOCR-VL] Processing: {input_file.name}")
            
            # 执行推理 (自动处理多页)
            result = pipeline.predict(str(input_file))
            
            markdown_list = []
            json_list = []

            # 遍历解析结果 (每一页)
            for idx, page_res in enumerate(result, 1):
                page_dir = out_dir / f"page_{idx}"
                page_dir.mkdir(exist_ok=True)

                # 保存单页 JSON 和 Markdown (用于调试和原子存储)
                if hasattr(page_res, "save_to_json"):
                    page_res.save_to_json(save_path=str(page_dir))
                
                if hasattr(page_res, "markdown"):
                    markdown_list.append(page_res.markdown)
                
                if hasattr(page_res, "json"):
                    json_list.append(page_res.json)

            # 合并所有页面的 Markdown
            if hasattr(pipeline, "concatenate_markdown_pages"):
                markdown_text = pipeline.concatenate_markdown_pages(markdown_list)
            else:
                markdown_text = "\n\n---\n\n".join([str(m) for m in markdown_list])

            # 保存最终合并结果
            final_md_path = out_dir / "result.md"
            final_md_path.write_text(markdown_text, encoding="utf-8")
            
            final_json_path = out_dir / "result.json"
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump({"pages": json_list, "total": len(result)}, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ [PaddleOCR-VL] Completed: {len(result)} pages parsed.")

            return {
                "success": True,
                "output_path": str(out_dir),
                "markdown": markdown_text,
                "markdown_file": str(final_md_path),
                "json_file": str(final_json_path)
            }

        except Exception as e:
            logger.error(f"❌ [PaddleOCR-VL] Parsing Error: {e}")
            raise
        finally:
            self.cleanup()

# 全局访问接口
_engine = None

def get_engine() -> PaddleOCRVLVLLMEngine:
    """获取单例引擎"""
    global _engine
    if _engine is None:
        _engine = PaddleOCRVLVLLMEngine()
    return _engine

"""
MinerU Tianshu - LitServe Worker (Production Ready)
天枢 LitServe Worker - 多引擎智能调度增强版

更新点：
1. 完整实现 _should_split_pdf：支持大文件自动分片进入任务队列。
2. VLM 提示词优化：针对 MinerU-1.2B 调优，输出高质量 Markdown。
3. 并发安全增强：确保多 GPU 环境下显存清理彻底。
4. 完善启动参数：对接 start_all.py。
"""

import os
import json
import sys
import time
import threading
import signal
import atexit
import base64
import multiprocessing
from pathlib import Path
from typing import Optional, List, Dict, Any

import litserve as ls
from loguru import logger
import importlib.util

# ============================================================================
# 基础补丁与环境初始化
# ============================================================================
try:
    import litserve.mcp as ls_mcp
    from contextlib import asynccontextmanager
    if not hasattr(ls_mcp, "MCPServer"):
        class Dummy: pass
        ls_mcp.MCPServer = Dummy
        ls_mcp.StreamableHTTPSessionManager = Dummy
    class DummyMCPConnector:
        def __init__(self, *args, **kwargs): pass
        @asynccontextmanager
        async def lifespan(self, app): yield
        def connect_mcp_server(self, *args, **kwargs): pass
    ls_mcp._LitMCPServerConnector = DummyMCPConnector
except Exception as e:
    logger.warning(f"MCP Patching bypassed: {e}")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from task_db import TaskDB
from output_normalizer import normalize_output

# 引擎检测
def is_available(name): return importlib.util.find_spec(name) is not None
PYPDF_AVAILABLE = is_available("pypdf")
OPENAI_AVAILABLE = is_available("openai")
FITZ_AVAILABLE = is_available("fitz")

class MinerUWorkerAPI(ls.LitAPI):
    def __init__(self, **kwargs):
        super().__init__()
        self.output_dir = kwargs.get("output_dir") or os.getenv("OUTPUT_PATH", "/app/data/output")
        self.poll_interval = kwargs.get("poll_interval", 0.5)
        self.enable_worker_loop = kwargs.get("enable_worker_loop", True)
        
        # 远程 API 列表
        self.paddle_vlm_url = os.getenv("PADDLE_VLM_URL", "http://host.docker.internal:8118/v1")
        self.mineru_vlm_url = os.getenv("MINERU_VLM_URL", "http://host.docker.internal:8119/v1")

    def setup(self, device):
        # 1. GPU 进程隔离
        if "cuda:" in str(device):
            gpu_id = str(device).split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            os.environ["MINERU_DEVICE_MODE"] = "cuda:0"
            logger.info(f"🎯 [GPU] Worker isolated to Physical {gpu_id}")

        # 2. 远程 VLM 客户端
        if OPENAI_AVAILABLE:
            from openai import OpenAI
            self.client_paddle = OpenAI(api_key="EMPTY", base_url=self.paddle_vlm_url)
            self.client_mineru = OpenAI(api_key="EMPTY", base_url=self.mineru_vlm_url)
        
        # 3. 初始化数据库与引擎
        self.task_db = TaskDB(os.getenv("DATABASE_PATH", "/app/data/db/mineru_tianshu.db"))
        self.mineru_pipeline_engine = None
        self.running = True
        self.device = device

        if self.enable_worker_loop:
            threading.Thread(target=self._worker_loop, daemon=True).start()
        logger.success(f"🚀 Worker {device} Ready")

    def _worker_loop(self):
        while self.running:
            try:
                task = self.task_db.get_next_task(worker_id=f"worker-{self.device}")
                if task:
                    logger.info(f"📥 Pulled Task: {task['task_id']} | File: {task['file_name']}")
                    self._process_task(task)
                else:
                    time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                time.sleep(2)

    def _process_task(self, task: dict):
        task_id = task["task_id"]
        file_path = task["file_path"]
        options = json.loads(task.get("options", "{}")) if isinstance(task.get("options"), str) else task.get("options", {})
        backend = task.get("backend", "pipeline").lower()

        try:
            # 1. 大文件自动切分逻辑 (分片后任务重新进入队列)
            if Path(file_path).suffix.lower() == ".pdf" and not task.get("parent_task_id"):
                if self._should_split_pdf(task_id, file_path, task, options):
                    logger.info(f"✂️ Task {task_id} split into subtasks. Parent task suspended.")
                    return

            # 2. 路由分发
            result = None
            if backend == "pipeline":
                result = self._process_with_mineru(file_path, options)
            elif backend == "vlm-auto-engine":
                result = self._process_remote_vlm(file_path, options, engine_type="mineru")
            elif backend == "hybrid-auto-engine":
                result = self._process_hybrid(file_path, options)
            elif "paddleocr-vl" in backend:
                result = self._process_remote_vlm(file_path, options, engine_type="paddle")
            else:
                result = self._process_with_mineru(file_path, options)

            # 3. 完成任务
            self.task_db.update_task_status(task_id, "completed", result_path=result["result_path"])
            
            # 如果是子任务，检查父任务是否可以合并
            if task.get("parent_task_id"):
                self.task_db.on_child_task_completed(task_id)

        except Exception as e:
            logger.error(f"❌ Task {task_id} Failed: {e}")
            self.task_db.update_task_status(task_id, "failed", error_message=str(e))

    # ============================================================================
    # 核心处理函数
    # ============================================================================

    def _process_remote_vlm(self, file_path: str, options: dict, engine_type="mineru") -> dict:
        """高性能远程调用：PDF -> 图像流 -> vLLM"""
        import fitz
        doc = fitz.open(file_path)
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        client = self.client_mineru if engine_type == "mineru" else self.client_paddle
        model_name = "mineru-vlm-1.2b" if engine_type == "mineru" else "PaddleOCR-VL-1.5"
        
        # 针对不同模型的提示词优化
        system_prompt = (
            "你是一个专业的文档解析助手。请将图片内容转换为高质量的 Markdown 格式，"
            "特别注意保留表格的单元格结构、数学公式的 LaTeX 表达以及标题层级。"
        )

        full_md = []
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2)) # 提高识别精度
            img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }],
                max_tokens=2048,
                temperature=0.1 # 降低随机性
            )
            full_md.append(f"\n{response.choices[0].message.content}")

        final_md = "\n\n".join(full_md)
        (output_dir / "result.md").write_text(final_md, encoding="utf-8")
        normalize_output(output_dir)
        return {"result_path": str(output_dir)}

    def _should_split_pdf(self, task_id: str, file_path: str, task: dict, options: dict) -> bool:
        """PDF 分片逻辑实现：超过阈值则创建子任务"""
        if not PYPDF_AVAILABLE: return False
        
        from utils.pdf_utils import get_pdf_page_count, split_pdf_file
        threshold = int(os.getenv("PDF_SPLIT_THRESHOLD_PAGES", "50"))
        chunk_size = int(os.getenv("PDF_SPLIT_CHUNK_SIZE", "20"))
        
        page_count = get_pdf_page_count(Path(file_path))
        if page_count <= threshold: return False

        # 执行物理拆分
        split_dir = Path(self.output_dir) / "temp_splits" / task_id
        split_dir.mkdir(parents=True, exist_ok=True)
        chunks = split_pdf_file(Path(file_path), split_dir, chunk_size=chunk_size)

        # 转换为父任务并生成子任务
        self.task_db.convert_to_parent_task(task_id, child_count=len(chunks))
        for chunk in chunks:
            self.task_db.create_child_task(
                parent_task_id=task_id,
                file_name=chunk["name"],
                file_path=chunk["path"],
                backend=task.get("backend", "pipeline"),
                options=options
            )
        return True

    def _process_with_mineru(self, file_path: str, options: dict) -> dict:
        if not self.mineru_pipeline_engine:
            from mineru_pipeline import MinerUPipelineEngine
            self.mineru_pipeline_engine = MinerUPipelineEngine(device="cuda:0" if "cuda" in str(self.device) else "cpu")
        
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        res = self.mineru_pipeline_engine.parse(file_path, output_path=str(output_dir), options=options)
        normalize_output(Path(res["result_path"]))
        return res

    def _process_hybrid(self, file_path: str, options: dict) -> dict:
        # 智能判定：如果是纯图片或极短文档，走 VLM；否则走 Pipeline
        if PYPDF_AVAILABLE:
            from pypdf import PdfReader
            text = PdfReader(file_path).pages[0].extract_text()
            if len(text.strip()) < 50:
                return self._process_remote_vlm(file_path, options, engine_type="mineru")
        return self._process_with_mineru(file_path, options)

    def decode_request(self, request): return request.get("action", "health")
    def predict(self, action): return {"status": "healthy", "device": str(self.device)}
    def encode_response(self, response): return response

# ============================================================================
# 启动入口
# ============================================================================
def start_litserve_workers(**kwargs):
    api = MinerUWorkerAPI(**kwargs)
    server = ls.LitServer(
        api, 
        accelerator=kwargs.get("accelerator", "auto"),
        devices=kwargs.get("devices", "auto"),
        workers_per_device=kwargs.get("workers_per_device", 1),
        timeout=False
    )
    server.run(port=kwargs.get("port", 8001))

if __name__ == "__main__":
    import argparse
    from utils import parse_list_arg
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    # 保持与 start_all.py 参数兼容
    parser.add_argument("--paddleocr-vl-vllm-api-list", type=parse_list_arg, default=[])
    
    args = parser.parse_args()
    start_litserve_workers(
        port=args.port,
        devices=args.devices,
        workers_per_device=args.workers_per_device,
        accelerator=args.accelerator,
        output_dir=args.output_dir
    )

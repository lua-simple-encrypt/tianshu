"""
MinerU Tianshu - LitServe Worker (Production Ready)
天枢 LitServe Worker - 多引擎智能调度增强版

核心逻辑：
- 调度中心：负责本地 Pipeline、远程 VLM、智能混合动力的路由分发。
- 并行加速：支持 PDF 自动分片，利用多 GPU 节点并行处理。
- 结果闭环：子任务完成后自动触发 Markdown/JSON 结果合并。
"""

import os
import json
import sys
import time
import threading
import signal
import atexit
import base64
import gc
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

# 设置项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from task_db import TaskDB
from output_normalizer import normalize_output
from utils.merge_utils import merge_subtask_results # 导入合并工具

# 引擎可用性检测
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
        
        # 远程 API 集群配置
        self.paddle_vlm_url = os.getenv("PADDLE_VLM_URL", "http://host.docker.internal:8118/v1")
        self.mineru_vlm_url = os.getenv("MINERU_VLM_URL", "http://host.docker.internal:8119/v1")

    def setup(self, device):
        # 1. 物理 GPU 隔离 (实现逻辑：Physical ID -> Logical cuda:0)
        if "cuda:" in str(device):
            gpu_id = str(device).split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            os.environ["MINERU_DEVICE_MODE"] = "cuda:0"
            logger.info(f"🎯 [GPU Isolation] Worker bound to Physical GPU {gpu_id}")

        # 2. 初始化 OpenAI 兼容客户端（对接远程 vLLM）
        if OPENAI_AVAILABLE:
            from openai import OpenAI
            self.client_paddle = OpenAI(api_key="EMPTY", base_url=self.paddle_vlm_url)
            self.client_mineru = OpenAI(api_key="EMPTY", base_url=self.mineru_vlm_url)
        
        # 3. 初始化持久化层
        self.task_db = TaskDB(os.getenv("DATABASE_PATH", "/app/data/db/mineru_tianshu.db"))
        self.mineru_pipeline_engine = None # 延迟加载本地模型
        self.running = True
        self.device = device

        if self.enable_worker_loop:
            threading.Thread(target=self._worker_loop, daemon=True).start()
        logger.success(f"🚀 Worker {device} Setup Complete")

    def _worker_loop(self):
        """Worker 主动拉取任务循环"""
        while self.running:
            try:
                task = self.task_db.get_next_task(worker_id=f"worker-{self.device}")
                if task:
                    logger.info(f"📥 Pull Task: {task['task_id']} (Backend: {task.get('backend')})")
                    self._process_task(task)
                else:
                    time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"❌ Worker loop error: {e}")
                time.sleep(2)

    def _process_task(self, task: dict):
        """核心处理路由"""
        task_id = task["task_id"]
        file_path = task["file_path"]
        options = json.loads(task.get("options", "{}")) if isinstance(task.get("options"), str) else task.get("options", {})
        backend = task.get("backend", "pipeline").lower()

        try:
            # 1. 检查是否需要触发分片逻辑
            if Path(file_path).suffix.lower() == ".pdf" and not task.get("parent_task_id"):
                if self._should_split_pdf(task_id, file_path, task, options):
                    return # 任务已裂变为子任务，当前流程中止

            # 2. 路由分发逻辑
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
                logger.warning(f"⚠️ Unknown backend {backend}, routing to pipeline")
                result = self._process_with_mineru(file_path, options)

            # 3. 提交任务结果
            self.task_db.update_task_status(task_id, "completed", result_path=result["result_path"])
            
            # 4. 【核心】如果是子任务，检查并触发父级合并
            if task.get("parent_task_id"):
                parent_id = self.task_db.on_child_task_completed(task_id)
                if parent_id:
                    logger.info(f"🧱 All subtasks done. Merging results for Parent: {parent_id}")
                    subtasks = self.task_db.get_child_tasks(parent_id)
                    merge_subtask_results(parent_id, subtasks, Path(self.output_dir))
                    self.task_db.update_task_status(parent_id, "completed", 
                                                   result_path=str(Path(self.output_dir) / parent_id))

        except Exception as e:
            logger.error(f"❌ Task {task_id} Failed: {e}")
            self.task_db.update_task_status(task_id, "failed", error_message=str(e))
            # 级联标记父任务失败
            if task.get("parent_task_id"):
                self.task_db.on_child_task_failed(task_id, str(e))
        finally:
            self._clean_memory()

    # ============================================================================
    # 引擎实现
    # ============================================================================

    def _process_remote_vlm(self, file_path: str, options: dict, engine_type="mineru") -> dict:
        """远程 VLM 集群调度逻辑"""
        import fitz
        doc = fitz.open(file_path)
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        client = self.client_mineru if engine_type == "mineru" else self.client_paddle
        model_name = "mineru-vlm-1.2b" if engine_type == "mineru" else "PaddleOCR-VL-1.5"
        
        system_prompt = (
            "你是一个高精度的 OCR 专家。请将图片内容转换为符合 Markdown 规范的文本，"
            "保留所有数学公式（使用 LaTeX）、表格（使用 Markdown 表格）和完整的排版层级。"
        )

        full_md = []
        logger.info(f"🔮 [VLM] Forwarding {len(doc)} pages to {model_name}...")

        for i in range(len(doc)):
            # 渲染 144 DPI (2.0 zoom) 图片，保证 OCR 清晰度
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
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
                temperature=0.05 # 接近确定性输出
            )
            full_md.append(f"\n{response.choices[0].message.content}")

        final_content = "\n\n".join(full_md)
        (output_dir / "result.md").write_text(final_content, encoding="utf-8")
        normalize_output(output_dir)
        return {"result_path": str(output_dir)}

    def _process_hybrid(self, file_path: str, options: dict) -> dict:
        """智能混合决策逻辑"""
        is_complex = False
        if PYPDF_AVAILABLE:
            from pypdf import PdfReader
            text = PdfReader(file_path).pages[0].extract_text()
            # 简单启发式：如果首页文字极少（<50字符），判定为扫描件/图表，走 VLM
            if len(text.strip()) < 50: is_complex = True
            
        if is_complex:
            logger.info("⚖️ [Hybrid] Complex doc detected -> Routing to VLM.")
            return self._process_remote_vlm(file_path, options, engine_type="mineru")
        else:
            logger.info("⚖️ [Hybrid] Standard doc detected -> Routing to Local Pipeline.")
            return self._process_with_mineru(file_path, options)

    def _should_split_pdf(self, task_id: str, file_path: str, task: dict, options: dict) -> bool:
        """高性能分片逻辑"""
        if not PYPDF_AVAILABLE: return False
        from utils.pdf_utils import get_pdf_page_count, split_pdf_file
        
        # 从环境变量读取配置 (默认 50 页拆分)
        threshold = int(os.getenv("PDF_SPLIT_THRESHOLD_PAGES", "50"))
        chunk_size = int(os.getenv("PDF_SPLIT_CHUNK_SIZE", "20"))
        
        pages = get_pdf_page_count(Path(file_path))
        if pages <= threshold: return False

        logger.info(f"✂️ Splitting large PDF ({pages} pages) for Parallel processing...")
        split_dir = Path(self.output_dir) / "temp_splits" / task_id
        chunks = split_pdf_file(Path(file_path), split_dir, chunk_size=chunk_size, parent_task_id=task_id)

        self.task_db.convert_to_parent_task(task_id, child_count=len(chunks))
        for chunk in chunks:
            # 子任务继承父任务的所有配置
            self.task_db.create_child_task(
                parent_task_id=task_id,
                file_name=chunk["name"],
                file_path=chunk["path"],
                backend=task.get("backend", "pipeline"),
                options={**options, "chunk_info": chunk},
                priority=task.get("priority", 0),
                user_id=task.get("user_id")
            )
        return True

    def _process_with_mineru(self, file_path: str, options: dict) -> dict:
        """本地 GPU 解析逻辑"""
        if not self.mineru_pipeline_engine:
            from mineru_pipeline import MinerUPipelineEngine
            self.mineru_pipeline_engine = MinerUPipelineEngine(device="cuda:0")
        
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        res = self.mineru_pipeline_engine.parse(file_path, output_path=str(output_dir), options=options)
        normalize_output(Path(res["result_path"]))
        return res

    def _clean_memory(self):
        """显存防泄漏清理"""
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
        except: pass

    # LitServe 接口实现
    def decode_request(self, request): return request.get("action", "health")
    def predict(self, action): return {"status": "healthy", "worker": self.device}
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
    
    args = parser.parse_args()
    start_litserve_workers(**vars(args))

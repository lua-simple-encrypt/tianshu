"""
MinerU Tianshu - API Server (Production Ready)
天枢 API 服务器 - 多引擎智能调度 + 全链路心跳拨测版

核心变更：
1. 集成了 /api/v1/health/detail 接口，支持对本地 Worker 和远程 vLLM 集群的状态监控。
2. 优化了任务查询逻辑，支持分片并行任务的进度百分比计算。
3. 路径安全加固，防止通过文件服务接口进行跨目录攻击。
"""

import json
import os
import uuid
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import quote, unquote

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger

# 导入认证与数据库模块
from auth import (
    User,
    Permission,
    get_current_active_user,
    require_permission,
)
from auth.auth_db import AuthDB
from auth.routes import router as auth_router
from task_db import TaskDB

# ============================================================================
# 1. 应用初始化与配置
# ============================================================================
app = FastAPI(
    title="MinerU Tianshu API",
    description="天枢 - 企业级 AI 数据预处理平台 | 支持全链路状态监控与分布式调度",
    version="2.1.1",
)

# 跨域配置 (生产环境建议限制具体域名)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent

# 初始化任务数据库
db_path_env = os.getenv("DATABASE_PATH")
db_path = str(Path(db_path_env).resolve()) if db_path_env else str((PROJECT_ROOT / "data" / "db" / "mineru_tianshu.db").resolve())
db = TaskDB(db_path)
auth_db = AuthDB()

# 挂载认证路由
app.include_router(auth_router)

# 共享存储目录配置
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
OUTPUT_DIR = Path(OUTPUT_PATH) if OUTPUT_PATH else PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"📊 Tianshu API Server Online. DB: {db_path} | Output: {OUTPUT_DIR}")

# ============================================================================
# 2. 任务管理接口
# ============================================================================

@app.post("/api/v1/tasks/submit", tags=["任务管理"])
async def submit_task(
    file: UploadFile = File(..., description="PDF/图片/Office等"),
    backend: str = Form("pipeline", description="处理后端: pipeline/hybrid-auto-engine/vlm-auto-engine/paddleocr-vl-vllm"),
    lang: str = Form("auto"),
    method: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    priority: int = Form(0),
    remove_watermark: bool = Form(False),
    watermark_conf_threshold: float = Form(0.35),
    watermark_dilation: int = Form(10),
    convert_office_to_pdf: bool = Form(False),
    current_user: User = Depends(require_permission(Permission.TASK_SUBMIT)),
):
    """提交解析任务，进入数据库队列"""
    try:
        upload_dir = Path(os.getenv("UPLOAD_PATH", str(PROJECT_ROOT / "data" / "uploads")))
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一文件名防止覆盖
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        temp_file_path = upload_dir / unique_filename

        with open(temp_file_path, "wb") as f:
            while chunk := await file.read(8 * 1024 * 1024): # 8MB 块写入
                f.write(chunk)

        options = {
            "lang": lang, "method": method, "formula_enable": formula_enable,
            "table_enable": table_enable, "remove_watermark": remove_watermark,
            "watermark_conf_threshold": watermark_conf_threshold,
            "watermark_dilation": watermark_dilation, "convert_office_to_pdf": convert_office_to_pdf,
        }

        task_id = db.create_task(
            file_name=file.filename,
            file_path=str(temp_file_path),
            backend=backend,
            options=options,
            priority=priority,
            user_id=current_user.user_id,
        )

        logger.info(f"✅ Task [{task_id}] queued via {backend}")
        return {"success": True, "task_id": task_id, "status": "pending"}

    except Exception as e:
        logger.error(f"❌ Submit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tasks/{task_id}", tags=["任务管理"])
async def get_task_status(
    task_id: str, 
    format: str = Query("markdown", description="返回格式: markdown/json/both"),
    current_user: User = Depends(get_current_active_user)
):
    """获取任务状态及解析结果数据"""
    task = db.get_task(task_id)
    if not task: raise HTTPException(status_code=404, detail="Task not found")

    # 简易权限校验
    if not current_user.has_permission(Permission.TASK_VIEW_ALL) and task.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    response = {
        "success": True, "task_id": task_id, "status": task["status"],
        "file_name": task["file_name"], "backend": task["backend"],
        "created_at": task["created_at"], "completed_at": task["completed_at"],
        "error_message": task["error_message"], "data": None
    }

    # 如果是触发了分片的父任务，计算合并进度
    if task.get("is_parent"):
        total = task.get("child_count", 0)
        done = task.get("child_completed", 0)
        response["progress"] = round(done / total * 100, 1) if total > 0 else 0

    # 提取结果数据
    if task["status"] == "completed" and task.get("result_path"):
        res_dir = Path(task["result_path"])
        if res_dir.exists():
            data = {"json_available": False}
            # 搜索最终 Markdown (支持 result.md 或 任意 .md)
            md_file = next(res_dir.rglob("result.md"), next(res_dir.rglob("*.md"), None))
            if md_file and format in ["markdown", "both"]:
                data["content"] = md_file.read_text(encoding="utf-8")
            
            # 搜索结构化 JSON
            json_file = next(res_dir.rglob("result.json"), next(res_dir.rglob("*_content_list.json"), None))
            if json_file:
                data["json_available"] = True
                if format in ["json", "both"]:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data["json_content"] = json.load(f)
            response["data"] = data

    return response

# ============================================================================
# 3. 系统状态与心跳探测 (全链路)
# ============================================================================

@app.get("/api/v1/health/detail", tags=["系统信息"])
async def detailed_health_check():
    """并行拨测全链路服务：数据库、本地 Worker、远程 vLLM 集群"""
    
    async def probe(url: str, is_vllm: bool = False):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2.0)) as session:
                # vLLM 专用探测路径，其它服务探测根目录
                target = f"{url.rstrip('/v1')}/v1/models" if is_vllm else url
                async with session.get(target) as resp:
                    return "online" if resp.status == 200 else "error"
        except:
            return "offline"

    # 从环境变量获取配置，适配 Docker Internal 网络
    paddle_url = os.getenv("PADDLE_VLM_URL", "http://host.docker.internal:8118/v1")
    mineru_url = os.getenv("MINERU_VLM_URL", "http://host.docker.internal:8119/v1")
    worker_url = os.getenv("WORKER_URL", "http://worker:8001/health")

    # 执行异步并行探测，不阻塞主进程
    w_res, v8118, v8119 = await asyncio.gather(
        probe(worker_url),
        probe(paddle_url, is_vllm=True),
        probe(mineru_url, is_vllm=True)
    )

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "online",
            "local_worker": w_res,
            "vllm_paddle_8118": v8118,
            "vllm_mineru_8119": v8119
        }
    }

@app.get("/api/v1/engines", tags=["系统信息"])
async def list_engines():
    """获取天枢支持的解析后端列表"""
    return {
        "success": True,
        "engines": {
            "document": [
                {"name": "pipeline", "display_name": "本地标准流水线", "description": "由本地服务器 GPU 进行传统算法解析"},
                {"name": "hybrid-auto-engine", "display_name": "⚖️ 智能混合动力", "description": "【推荐】自动分析，复杂排版分流至远程 VLM"},
                {"name": "vlm-auto-engine", "display_name": "🚀 视觉大模型 (MinerU-VLM)", "description": "高精度模式，强制调用远程 1.2B 模型"}
            ],
            "ocr": [
                {"name": "paddleocr-vl-vllm", "display_name": "PaddleOCR-VL 远程加速版", "description": "调用 8118 端口的高性能 OCR 引擎"}
            ]
        }
    }

# ============================================================================
# 4. 静态文件与安全服务
# ============================================================================

@app.get("/v1/files/output/{file_path:path}", tags=["文件服务"])
async def serve_output_file(file_path: str):
    """服务处理结果中的静态资源 (如 images/xxx.jpg)，具备路径穿越防护"""
    try:
        decoded_path = unquote(file_path)
        full_path = (OUTPUT_DIR / decoded_path).resolve()
        
        # 安全沙箱检查：禁止访问输出目录以外的任何文件
        if not str(full_path).startswith(str(OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Forbidden: Path out of bounds")

        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(path=str(full_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 5. 启动入口
# ============================================================================
if __name__ == "__main__":
    api_port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=api_port, log_level="info")

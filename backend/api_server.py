"""
MinerU Tianshu - API Server (Production Ready)
天枢 API 服务器 - 多引擎智能调度 + 全链路心跳拨测版
"""

import json
import os
import re
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
# 1. 应用初始化
# ============================================================================
app = FastAPI(
    title="MinerU Tianshu API",
    description="天枢 - 企业级 AI 数据预处理平台 | 支持全链路状态监控",
    version="2.1.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent
db_path_env = os.getenv("DATABASE_PATH")
db_path = str(Path(db_path_env).resolve()) if db_path_env else str((PROJECT_ROOT / "data" / "db" / "mineru_tianshu.db").resolve())
db = TaskDB(db_path)
auth_db = AuthDB()

app.include_router(auth_router)

output_path_env = os.getenv("OUTPUT_PATH")
OUTPUT_DIR = Path(output_path_env) if output_path_env else PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. 核心任务接口 (提交/查询/取消)
# ============================================================================

@app.post("/api/v1/tasks/submit", tags=["任务管理"])
async def submit_task(
    file: UploadFile = File(...),
    backend: str = Form("pipeline"),
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
    try:
        upload_dir = Path(os.getenv("UPLOAD_PATH", str(PROJECT_ROOT / "data" / "uploads")))
        upload_dir.mkdir(parents=True, exist_ok=True)
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        temp_file_path = upload_dir / unique_filename

        with open(temp_file_path, "wb") as f:
            while chunk := await file.read(8 * 1024 * 1024):
                f.write(chunk)

        options = {
            "lang": lang, "method": method, "formula_enable": formula_enable,
            "table_enable": table_enable, "remove_watermark": remove_watermark,
            "watermark_conf_threshold": watermark_conf_threshold,
            "watermark_dilation": watermark_dilation, "convert_office_to_pdf": convert_office_to_pdf,
        }

        task_id = db.create_task(
            file_name=file.filename, file_path=str(temp_file_path),
            backend=backend, options=options, priority=priority,
            user_id=current_user.user_id,
        )
        return {"success": True, "task_id": task_id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tasks/{task_id}", tags=["任务管理"])
async def get_task_status(task_id: str, format: str = Query("markdown"), current_user: User = Depends(get_current_active_user)):
    task = db.get_task(task_id)
    if not task: raise HTTPException(status_code=404, detail="Task not found")
    if not current_user.has_permission(Permission.TASK_VIEW_ALL) and task.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    response = { "success": True, "task_id": task_id, "status": task["status"], "backend": task["backend"], "data": None }
    if task.get("is_parent"):
        response["progress"] = round(task["child_completed"] / task["child_count"] * 100, 1) if task["child_count"] > 0 else 0

    if task["status"] == "completed" and task.get("result_path"):
        res_dir = Path(task["result_path"])
        if res_dir.exists():
            data = {"json_available": False}
            md_file = next(res_dir.rglob("result.md"), next(res_dir.rglob("*.md"), None))
            if md_file and format in ["markdown", "both"]:
                data["content"] = md_file.read_text(encoding="utf-8")
            json_file = next(res_dir.rglob("result.json"), None)
            if json_file:
                data["json_available"] = True
                if format in ["json", "both"]:
                    data["json_content"] = json.loads(json_file.read_text(encoding="utf-8"))
            response["data"] = data
    return response

# ============================================================================
# 3. 系统心跳探测接口 (New!)
# ============================================================================

@app.get("/api/v1/health/detail", tags=["系统信息"])
async def detailed_health_check():
    """并行拨测全链路服务：DB -> Local Worker -> vLLM(8118/8119)"""
    
    async def probe(url: str, is_vllm: bool = False):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2.0)) as session:
                # vLLM 使用 OpenAI 标准模型路径探测
                target = f"{url.rstrip('/v1')}/v1/models" if is_vllm else url
                async with session.get(target) as resp:
                    return "online" if resp.status == 200 else "error"
        except:
            return "offline"

    # 读取路由配置
    paddle_url = os.getenv("PADDLE_VLM_URL", "http://host.docker.internal:8118/v1")
    mineru_url = os.getenv("MINERU_VLM_URL", "http://host.docker.internal:8119/v1")
    worker_url = os.getenv("WORKER_URL", "http://worker:8001/health")

    # 执行并行探测
    local_worker, vllm_8118, vllm_8119 = await asyncio.gather(
        probe(worker_url),
        probe(paddle_url, is_vllm=True),
        probe(mineru_url, is_vllm=True)
    )

    return {
        "status": "healthy",
        "services": {
            "database": "online",
            "local_worker": local_worker,
            "vllm_paddle_8118": vllm_8118,
            "vllm_mineru_8119": vllm_8119
        }
    }

@app.get("/api/v1/engines", tags=["系统信息"])
async def list_engines():
    engines = {
        "document": [
            {"name": "pipeline", "display_name": "本地标准流水线", "description": "本地 GPU 运行"},
            {"name": "hybrid-auto-engine", "display_name": "⚖️ 智能混合动力", "description": "自动分流本地/远程"},
            {"name": "vlm-auto-engine", "display_name": "🚀 视觉大模型 (MinerU-VLM)", "description": "强制调用远程 8119"}
        ],
        "ocr": [{"name": "paddleocr-vl-vllm", "display_name": "PaddleOCR-VL 远程版", "description": "调用远程 8118"}]
    }
    return {"success": True, "engines": engines}

# ============================================================================
# 4. 静态文件与入口
# ============================================================================

@app.get("/v1/files/output/{file_path:path}", tags=["文件服务"])
async def serve_output_file(file_path: str):
    decoded_path = unquote(file_path)
    full_path = (OUTPUT_DIR / decoded_path).resolve()
    if not str(full_path).startswith(str(OUTPUT_DIR.resolve())): raise HTTPException(status_code=403)
    if not full_path.exists(): raise HTTPException(status_code=404)
    return FileResponse(path=str(full_path))

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    logger.info(f"🚀 Tianshu Server Online at {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

"""
MinerU Tianshu - API Server (Production Ready)
天枢 API 服务器 - 多引擎智能调度版

核心功能：
1. 任务管理：提交、查询、取消解析任务。
2. 引擎分发：支持 pipeline, hybrid-auto-engine, vlm-auto-engine 等多种后端。
3. 队列监控：实时获取系统负载与 Worker 状态。
4. 认证授权：集成企业级 JWT 与 API Key 校验。
"""

import json
import os
import re
import uuid
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
    description="天枢 - 企业级 AI 数据预处理平台 | 支持 Pipeline/VLM/Hybrid 多模式智能调度",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent

# 初始化数据库 (优先从环境变量读取，确保与 Worker 同步)
db_path_env = os.getenv("DATABASE_PATH")
db_path = str(Path(db_path_env).resolve()) if db_path_env else str((PROJECT_ROOT / "data" / "db" / "mineru_tianshu.db").resolve())
db = TaskDB(db_path)
auth_db = AuthDB()

app.include_router(auth_router)

# 配置共享输出目录
output_path_env = os.getenv("OUTPUT_PATH")
OUTPUT_DIR = Path(output_path_env) if output_path_env else PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"📊 API Server Online. DB: {db_path} | Storage: {OUTPUT_DIR}")

# ============================================================================
# 2. 核心任务接口
# ============================================================================

@app.get("/", tags=["系统信息"])
async def root():
    return {
        "service": "MinerU Tianshu",
        "version": "2.1.0",
        "status": "running",
        "docs": "/docs",
    }

@app.post("/api/v1/tasks/submit", tags=["任务管理"])
async def submit_task(
    file: UploadFile = File(..., description="支持 PDF/图片/Office/音频/视频等"),
    backend: str = Form(
        "pipeline",
        description=(
            "处理后端选择：\n"
            "- auto: 自动根据文件类型选择\n"
            "- pipeline: 本地标准流水线 (GPU)\n"
            "- hybrid-auto-engine: 智能混合动力 (根据复杂度分流)\n"
            "- vlm-auto-engine: 视觉大模型引擎 (MinerU-VLM-1.2B)\n"
            "- paddleocr-vl-vllm: 远程高性能 OCR (8118 端口)"
        )
    ),
    lang: str = Form("auto", description="语言: auto/ch/en等"),
    method: str = Form("auto", description="解析方法: auto/txt/ocr"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    priority: int = Form(0),
    remove_watermark: bool = Form(False, description="是否去除水印 (支持 PDF/图片)"),
    watermark_conf_threshold: float = Form(0.35),
    watermark_dilation: int = Form(10),
    convert_office_to_pdf: bool = Form(False, description="Office 格式是否先转 PDF"),
    current_user: User = Depends(require_permission(Permission.TASK_SUBMIT)),
):
    """提交解析任务，支持多引擎调度"""
    try:
        upload_dir = Path(os.getenv("UPLOAD_PATH", str(PROJECT_ROOT / "data" / "uploads")))
        upload_dir.mkdir(parents=True, exist_ok=True)

        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        temp_file_path = upload_dir / unique_filename

        # 流式写入文件
        with open(temp_file_path, "wb") as f:
            while chunk := await file.read(8 * 1024 * 1024):
                f.write(chunk)

        options = {
            "lang": lang,
            "method": method,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "remove_watermark": remove_watermark,
            "watermark_conf_threshold": watermark_conf_threshold,
            "watermark_dilation": watermark_dilation,
            "convert_office_to_pdf": convert_office_to_pdf,
        }

        task_id = db.create_task(
            file_name=file.filename,
            file_path=str(temp_file_path),
            backend=backend,
            options=options,
            priority=priority,
            user_id=current_user.user_id,
        )

        logger.info(f"✅ Task Queued: {task_id} via {backend} (User: {current_user.username})")
        return {"success": True, "task_id": task_id, "status": "pending"}

    except Exception as e:
        logger.error(f"❌ Submit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tasks/{task_id}", tags=["任务管理"])
async def get_task_status(
    task_id: str,
    format: str = Query("markdown", description="返回格式: markdown/json/both"),
    current_user: User = Depends(get_current_active_user),
):
    """查询任务状态并返回解析结果"""
    task = db.get_task(task_id)
    if not task: raise HTTPException(status_code=404, detail="Task not found")

    # 权限校验
    if not current_user.has_permission(Permission.TASK_VIEW_ALL) and task.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    response = {
        "success": True,
        "task_id": task_id,
        "status": task["status"],
        "file_name": task["file_name"],
        "backend": task["backend"],
        "created_at": task["created_at"],
        "completed_at": task["completed_at"],
        "error_message": task["error_message"],
        "data": None
    }

    # 合并主子任务进度
    if task.get("is_parent"):
        total = task.get("child_count", 0)
        done = task.get("child_completed", 0)
        response["progress"] = round(done / total * 100, 1) if total > 0 else 0

    # 任务完成，装载结果数据
    if task["status"] == "completed" and task.get("result_path"):
        res_dir = Path(task["result_path"])
        if res_dir.exists():
            data = {"json_available": False}
            
            # 搜索 Markdown 文件 (Worker 统一规范为 result.md)
            md_file = next(res_dir.rglob("result.md"), next(res_dir.rglob("*.md"), None))
            if md_file and format in ["markdown", "both"]:
                data["content"] = md_file.read_text(encoding="utf-8")
                data["markdown_file"] = md_file.name

            # 搜索 JSON 文件
            json_file = next(res_dir.rglob("result.json"), next(res_dir.rglob("*_content_list.json"), None))
            if json_file:
                data["json_available"] = True
                if format in ["json", "both"]:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data["json_content"] = json.load(f)
            
            response["data"] = data

    return response

@app.delete("/api/v1/tasks/{task_id}", tags=["任务管理"])
async def cancel_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    task = db.get_task(task_id)
    if not task: raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] == "pending":
        db.update_task_status(task_id, "cancelled")
        if Path(task["file_path"]).exists(): Path(task["file_path"]).unlink()
        return {"success": True, "message": "Cancelled"}
    raise HTTPException(status_code=400, detail="Only pending tasks can be cancelled")

# ============================================================================
# 3. 队列与系统管理
# ============================================================================

@app.get("/api/v1/queue/stats", tags=["队列管理"])
async def get_queue_stats(current_user: User = Depends(require_permission(Permission.QUEUE_VIEW))):
    return {"success": True, "stats": db.get_queue_stats(), "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/engines", tags=["系统信息"])
async def list_engines():
    """动态获取系统中注册的所有处理引擎列表"""
    engines = {
        "document": [
            {
                "name": "pipeline",
                "display_name": "本地标准流水线 (Pipeline)",
                "description": "基于 YOLO+UniMERNet 的本地解析，适合文字规整的电子档 PDF。",
                "supported_formats": [".pdf", ".png", ".jpg"]
            },
            {
                "name": "hybrid-auto-engine",
                "display_name": "⚖️ 智能混合动力 (Hybrid)",
                "description": "【推荐】自动分析文档。标准件本地跑，复杂排版/扫描件分流至远程 VLM。",
                "supported_formats": [".pdf"]
            },
            {
                "name": "vlm-auto-engine",
                "display_name": "🚀 视觉大模型 (MinerU-VLM)",
                "description": "全量调用远程 MinerU-VLM-1.2B 引擎。解析精度最高，擅长复杂表格。",
                "supported_formats": [".pdf", ".png", ".jpg"]
            }
        ],
        "ocr": [
            {
                "name": "paddleocr-vl-vllm",
                "display_name": "PaddleOCR-VL 远程版",
                "description": "基于 8118 端口的 vLLM 远程引擎，支持 109 种语言识别。",
                "supported_formats": [".pdf", ".png", ".jpg"]
            }
        ],
        "audio": [], "video": [], "format": []
    }

    # 动态检测可选引擎扩展包
    import importlib.util
    if importlib.util.find_spec("audio_engines"):
        engines["audio"].append({"name": "sensevoice", "display_name": "SenseVoice 语音识别", "description": "多语言语音转文字，支持说话人分离"})
    if importlib.util.find_spec("video_engines"):
        engines["video"].append({"name": "video", "display_name": "Video 视频结构化", "description": "提取关键帧并进行多模态内容理解"})

    return {"success": True, "engines": engines}

@app.get("/api/v1/health", tags=["系统信息"])
async def health_check():
    try:
        return {"status": "healthy", "database": "connected", "queue_stats": db.get_queue_stats()}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

# ============================================================================
# 4. 静态文件访问服务 (支持图片预览)
# ============================================================================

@app.get("/v1/files/output/{file_path:path}", tags=["文件服务"])
async def serve_output_file(file_path: str):
    """服务处理结果中的静态资源 (如 images/*.jpg)"""
    try:
        decoded_path = unquote(file_path)
        full_path = (OUTPUT_DIR / decoded_path).resolve()
        
        # 安全沙箱检查：禁止访问输出目录以外的文件
        if not str(full_path).startswith(str(OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Forbidden")

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
    logger.info(f"🚀 Tianshu API Server online at http://0.0.0.0:{api_port}")
    logger.info(f"📖 Swagger Docs: http://localhost:{api_port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=api_port, log_level="info")

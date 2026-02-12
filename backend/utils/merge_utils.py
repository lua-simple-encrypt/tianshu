"""
天枢任务合并工具 (Production Version)
功能：将 PDF 分片处理后的多个子任务结果（Markdown/JSON）完美无缝合并。
特性：支持页码修正、元数据聚合、异步 IO 安全。
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

def merge_subtask_results(
    parent_task_id: str, 
    subtasks: List[Dict[str, Any]], 
    output_dir: Path
) -> Dict[str, Any]:
    """
    聚合所有子任务的结果文件
    
    Args:
        parent_task_id: 父任务 ID
        subtasks: 数据库中查出的子任务列表（需包含 result_path 和 options）
        output_dir: 合并结果的存储目录
    """
    logger.info(f"🧩 Starting merge for parent task: {parent_task_id} ({len(subtasks)} chunks)")
    
    # 1. 按照起始页码对子任务进行物理排序
    def get_start_page(task):
        try:
            # options 可能是字符串或字典，取决于数据库驱动返回类型
            opts = task.get("options", {})
            if isinstance(opts, str):
                opts = json.loads(opts)
            return opts.get("chunk_info", {}).get("start_page", 0)
        except Exception:
            return 0

    sorted_tasks = sorted(subtasks, key=get_start_page)
    
    final_markdown = []
    final_json_pages = []
    total_images_copied = 0
    
    # 创建父任务的正式输出目录
    parent_res_dir = output_dir / parent_task_id
    parent_res_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建统一的 images 目录
    final_image_dir = parent_res_dir / "images"
    final_image_dir.mkdir(exist_ok=True)

    # 2. 顺序迭代处理每个分片
    for task in sorted_tasks:
        if task["status"] != "completed" or not task.get("result_path"):
            continue
            
        chunk_path = Path(task["result_path"])
        if not chunk_path.exists():
            logger.warning(f"⚠️ Result path for subtask {task['task_id']} missing: {chunk_path}")
            continue

        # --- A. 合并 Markdown ---
        # 优先查找 result.md，其次查找目录下的任意 .md
        md_file = next(chunk_path.rglob("result.md"), next(chunk_path.rglob("*.md"), None))
        if md_file:
            content = md_file.read_text(encoding="utf-8")
            # 添加分片注释，方便排查
            chunk_info = json.loads(task["options"]).get("chunk_info", {})
            marker = f"\n\n\n"
            final_markdown.append(marker + content)

        # --- B. 合并并修正 JSON ---
        json_file = next(chunk_path.rglob("result.json"), next(chunk_path.rglob("*_content_list.json"), None))
        if json_file:
            try:
                chunk_data = json.loads(json_file.read_text(encoding="utf-8"))
                # 如果是 MinerU 格式，数据在 'pages' 列表里
                pages = chunk_data.get("pages", [])
                
                # 计算页码偏移量
                # 如果分片 2 是从第 51 页开始，offset 就是 50
                offset = get_start_page(task) - 1
                
                for page in pages:
                    if "page_idx" in page:
                        page["page_idx"] += offset
                    if "page_number" in page:
                        page["page_number"] += offset
                    final_json_pages.append(page)
            except Exception as e:
                logger.error(f"❌ Failed to parse JSON for chunk {task['task_id']}: {e}")

        # --- C. 迁移本地图片 (如果有) ---
        # 注意：如果图片已上传 RustFS，Markdown 里已经是 URL，这里只需迁移未上传的本地备份
        chunk_image_dir = chunk_path / "images"
        if chunk_image_dir.exists():
            for img in chunk_image_dir.iterdir():
                if img.is_file():
                    shutil.copy2(img, final_image_dir / img.name)
                    total_images_copied += 1

    # 3. 写入最终文件
    final_md_path = parent_res_dir / "result.md"
    final_md_path.write_text("\n\n".join(final_markdown), encoding="utf-8")
    
    final_json_path = parent_res_dir / "result.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "parent_task_id": parent_task_id,
            "total_chunks": len(subtasks),
            "pages": final_json_pages
        }, f, ensure_ascii=False, indent=2)

    logger.success(f"✅ Merge complete: {parent_task_id}")
    logger.info(f"   - Final Markdown: {final_md_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"   - Final JSON: {len(final_json_pages)} pages")
    logger.info(f"   - Total Images: {total_images_copied}")

    return {
        "result_path": str(parent_res_dir),
        "markdown_path": str(final_md_path),
        "json_path": str(final_json_path)
    }

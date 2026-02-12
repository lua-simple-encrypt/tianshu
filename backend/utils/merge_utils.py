"""
天枢任务合并工具 (Production Version)
功能：将 PDF 分片处理后的多个子任务结果（Markdown/JSON）完美无缝合并。
特性：支持全局页码修正、图片资产聚合、Options 自动解析。
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
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
        subtasks: 数据库中查出的子任务列表（需包含 result_path, options, status）
        output_dir: 合并结果的根存储目录
    """
    logger.info(f"🧩 开始合并父任务: {parent_task_id} (分片数: {len(subtasks)})")
    
    # 1. 解析 Options 并按起始页码对子任务进行物理排序
    def get_start_page(task):
        try:
            opts = task.get("options", {})
            # 兼容数据库返回字符串的情况
            if isinstance(opts, str):
                opts = json.loads(opts)
            return opts.get("chunk_info", {}).get("start_page", 0)
        except Exception as e:
            logger.error(f"❌ 解析子任务 Options 失败: {e}")
            return 0

    # 过滤掉未完成或无结果的分片，并排序
    valid_subtasks = [t for t in subtasks if t.get("status") == "completed" and t.get("result_path")]
    sorted_tasks = sorted(valid_subtasks, key=get_start_page)
    
    if not sorted_tasks:
        raise ValueError(f"父任务 {parent_task_id} 下没有可用的已完成子任务结果")

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
        chunk_path = Path(task["result_path"])
        if not chunk_path.exists():
            logger.warning(f"⚠️ 子任务 {task['task_id']} 结果路径丢失: {chunk_path}")
            continue

        # --- A. 合并 Markdown ---
        # 寻找分片内的 md 文件
        md_file = next(chunk_path.rglob("result.md"), next(chunk_path.rglob("*.md"), None))
        if md_file:
            content = md_file.read_text(encoding="utf-8")
            # 插入分片占位符，防止段落粘连
            start_pg = get_start_page(task)
            final_markdown.append(f"\n\n\n\n" + content)

        # --- B. 合并并修正 JSON (核心难点) ---
        json_file = next(chunk_path.rglob("result.json"), next(chunk_path.rglob("*_content_list.json"), None))
        if json_file:
            try:
                chunk_data = json.loads(json_file.read_text(encoding="utf-8"))
                # 支持 MinerU 2.x 的 pages 结构
                pages = chunk_data.get("pages", [])
                
                # 计算全局偏移量 (例如：第二分片从51页开始，offset=50)
                offset = get_start_page(task) - 1
                
                for page in pages:
                    # 修正页码索引
                    if "page_idx" in page:
                        page["page_idx"] += offset
                    if "page_number" in page:
                        page["page_number"] += offset
                    # 修正层级结构中的子页码引用（如果有）
                    final_json_pages.append(page)
            except Exception as e:
                logger.error(f"❌ 修正分片 JSON 索引失败 {task['task_id']}: {e}")

        # --- C. 迁移本地图片资产 ---
        # 子任务的图片目录通常在 chunk_path/images/
        chunk_image_dir = chunk_path / "images"
        if chunk_image_dir.exists():
            for img in chunk_image_dir.iterdir():
                if img.is_file():
                    # 这里使用 copy2 保留元数据，防止重名直接覆盖
                    target_img = final_image_dir / img.name
                    if not target_img.exists():
                        shutil.copy2(img, target_img)
                        total_images_copied += 1

    # 3. 持久化合并结果
    final_md_path = parent_res_dir / "result.md"
    # 使用 3 个换行符确保分片间有清晰的视觉间隔
    final_md_path.write_text("\n\n\n".join(final_markdown), encoding="utf-8")
    
    final_json_path = parent_res_dir / "result.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "parent_task_id": parent_task_id,
            "total_pages": len(final_json_pages),
            "merged_chunks": len(sorted_tasks),
            "pages": final_json_pages
        }, f, ensure_ascii=False, indent=2)

    logger.success(f"✅ 任务合并完成: {parent_task_id}")
    logger.info(f"   - Markdown 大小: {final_md_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"   - JSON 页数: {len(final_json_pages)}")
    logger.info(f"   - 聚合图片数: {total_images_copied}")

    return {
        "result_path": str(parent_res_dir),
        "markdown_path": str(final_md_path),
        "json_path": str(final_json_path)
    }

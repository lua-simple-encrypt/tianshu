"""
天枢 PDF 处理工具库 (Production Version)
集成了页数检测、物理拆分及高清图片转换功能。
依赖：PyMuPDF (fitz), pypdf, pikepdf
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

def get_pdf_page_count(pdf_path: Path) -> int:
    """
    快速获取 PDF 总页数 (基于 pypdf)
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except ImportError:
        logger.error("❌ pypdf not installed. Run: pip install pypdf")
        raise RuntimeError("pypdf is required")
    except Exception as e:
        logger.error(f"❌ Failed to read PDF page count: {e}")
        return 0

def convert_pdf_to_images(
    pdf_path: Path, 
    output_dir: Path, 
    zoom: float = 2.0, 
    dpi: Optional[int] = None
) -> List[Path]:
    """
    将 PDF 所有页高清转换为图片 (供 VLM 引擎调用)
    
    Args:
        zoom: 缩放比例，默认 2.0 (144 DPI)，适合 OCR
        dpi: 若指定则覆盖 zoom
    """
    try:
        import fitz  # PyMuPDF
        output_dir.mkdir(parents=True, exist_ok=True)
        
        doc = fitz.open(str(pdf_path))
        image_paths = []
        
        # 确定缩放矩阵
        scale = dpi / 72.0 if dpi else zoom
        mat = fitz.Matrix(scale, scale)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)
            
            # 命名规范：{文件名}_page_001.png
            img_name = f"{pdf_path.stem}_page_{page_num + 1:03d}.png"
            img_path = output_dir / img_name
            
            pix.save(str(img_path))
            image_paths.append(img_path)
            
        doc.close()
        logger.info(f"✅ Converted {len(image_paths)} pages to images in {output_dir}")
        return image_paths

    except Exception as e:
        logger.error(f"❌ PDF to Image conversion failed: {e}")
        raise

def split_pdf_file(
    pdf_path: Path, 
    output_dir: Path, 
    chunk_size: int = 50, 
    parent_task_id: str = "task"
) -> List[Dict[str, Any]]:
    """
    高性能拆分 PDF (基于 pikepdf 引用复制)
    支持大文件快速分片，为并行处理做准备。

    Returns:
        List[Dict]: 包含 path, name, start_page, end_page, page_count
    """
    try:
        import pikepdf
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            chunks = []
            
            logger.info(f"✂️ Splitting {pdf_path.name} ({total_pages} pages) into chunks of {chunk_size}")

            for i in range(0, total_pages, chunk_size):
                start_idx = i
                end_idx = min(i + chunk_size, total_pages)
                
                # 创建新 PDF 容器
                new_pdf = pikepdf.new()
                new_pdf.pages.extend(pdf.pages[start_idx:end_idx])
                
                # 文件名：{task_id}_chunk_001-050.pdf
                chunk_name = f"{parent_task_id}_chunk_{start_idx+1:03d}-{end_idx:03d}.pdf"
                chunk_path = output_dir / chunk_name
                
                new_pdf.save(chunk_path)
                
                chunks.append({
                    "name": chunk_name,
                    "path": str(chunk_path),
                    "start_page": start_idx + 1,
                    "end_page": end_idx,
                    "page_count": end_idx - start_idx
                })

            logger.info(f"✅ Successfully created {len(chunks)} PDF chunks.")
            return chunks

    except ImportError:
        logger.error("❌ pikepdf not installed. Run: pip install pikepdf")
        raise
    except Exception as e:
        logger.error(f"❌ PDF Split failed: {e}")
        raise

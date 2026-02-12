#!/usr/bin/env python3
"""
Tianshu Model Downloader - Offline Preparation Script
=====================================================
Downloads all required models for MinerU, PaddleOCR, PaddleX, and SenseVoice
to prepare for offline/air-gapped deployment.

Usage:
    python download_models.py --output ./models
    python download_models.py --output ./models --models mineru,paddlex
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from loguru import logger

# Configure Logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# ==============================================================================
# Model Configurations
# ==============================================================================
MODELS = {
    # 1. MinerU / PDF-Extract-Kit
    "mineru": {
        "name": "MinerU PDF-Extract-Kit 1.0",
        "repo_id": "opendatalab/PDF-Extract-Kit-1.0",
        "source": "huggingface",
        "target_dir": "mineru_cache/models",  # Maps to /root/.cache/mineru
        "description": "Core PDF extraction models (Layout, Formula, Reading Order)",
        "required": True
    },
    
    # 2. DocLayout-YOLO (Required by MinerU 2.7.6)
    "doclayout": {
        "name": "DocLayout-YOLO",
        "repo_id": "opendatalab/DocLayout-YOLO",
        "source": "huggingface",
        "target_dir": "mineru_cache/models/DocLayout-YOLO",
        "description": "Advanced layout detection for MinerU",
        "required": True
    },

    # 3. PaddleOCR / PaddleX Official Models
    # Note: PaddleX downloads models on-the-fly, but we pre-download the core set
    "paddlex_ocr": {
        "name": "PaddleOCR v4 Server Models",
        "urls": [
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar",
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar",
            "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
        ],
        "source": "direct",
        "target_dir": "paddlex_cache/official_models", # Maps to /root/.paddlex/official_models
        "description": "High-accuracy OCR models for server deployment",
        "required": True
    },

    # 4. SenseVoice (Speech-to-Text)
    "sensevoice": {
        "name": "SenseVoice Small",
        "model_id": "iic/SenseVoiceSmall",
        "source": "modelscope",
        "target_dir": "modelscope_cache/sensevoice",
        "description": "Multi-language high-speed speech recognition",
        "required": True
    },

    # 5. Paraformer (Speaker Diarization)
    "paraformer": {
        "name": "Paraformer Large (Speaker Diarization)",
        "model_id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "source": "modelscope",
        "target_dir": "modelscope_cache/paraformer",
        "description": "Speaker verification and diarization",
        "required": False
    },

    # 6. YOLO11 (Watermark Detection)
    "yolo11": {
        "name": "YOLO11x Watermark Detection",
        "repo_id": "corzent/yolo11x_watermark_detection",
        "filename": "best.pt",
        "source": "huggingface",
        "target_dir": "watermark_models",
        "description": "Custom YOLO model for detecting watermarks",
        "required": False
    },
    
    # 7. LaMa (Inpainting) - usually handled by simple-lama-inpainting, 
    # but we download it here to be safe for offline use.
    "lama": {
        "name": "Big LaMa Inpainting",
        "urls": [
            "https://github.com/advimman/lama/releases/download/resolutions/big-lama.zip"
        ],
        "source": "direct",
        "target_dir": "cache/torch/hub/checkpoints", # Standard torch hub location
        "description": "Image inpainting model for watermark removal",
        "required": False
    }
}

# ==============================================================================
# Helper Functions
# ==============================================================================

def download_file_direct(url, target_dir):
    """Download file from direct URL"""
    import requests
    import tarfile
    import zipfile
    
    filename = url.split("/")[-1]
    save_path = Path(target_dir) / filename
    
    if save_path.exists():
        logger.info(f"    ℹ️  File exists: {filename}")
        return save_path

    logger.info(f"    ⬇️  Downloading: {url}")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract if tar/zip
        if filename.endswith(".tar"):
            logger.info(f"    📦 Extracting tar: {filename}")
            with tarfile.open(save_path) as tar:
                tar.extractall(path=target_dir)
        elif filename.endswith(".zip"):
            logger.info(f"    📦 Extracting zip: {filename}")
            with zipfile.open(save_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
                
        return save_path
    except Exception as e:
        logger.error(f"    ❌ Download failed: {e}")
        return None

def download_from_huggingface(repo_id, target_dir, filename=None):
    """Download from HuggingFace Hub"""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download

        # Use HF Mirror for China
        hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        if filename:
            logger.info(f"    ⬇️  Downloading file: {filename}")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False, # Important for offline copy
                resume_download=True
            )
        else:
            logger.info(f"    ⬇️  Downloading repo: {repo_id}")
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False, # Important for offline copy
                resume_download=True,
                ignore_patterns=["*.git*", "*.md"] # Skip git files
            )
        return path
    except ImportError:
        logger.error("    ❌ huggingface_hub not installed. Run: pip install huggingface-hub")
        return None
    except Exception as e:
        logger.error(f"    ❌ HuggingFace download failed: {e}")
        return None

def download_from_modelscope(model_id, target_dir):
    """Download from ModelScope"""
    try:
        from modelscope import snapshot_download
        logger.info(f"    ⬇️  Downloading from ModelScope: {model_id}")
        path = snapshot_download(
            model_id,
            cache_dir=str(Path(target_dir).parent), # Modelscope creates its own subdir
            revision="master"
        )
        return path
    except ImportError:
        logger.error("    ❌ modelscope not installed. Run: pip install modelscope")
        return None
    except Exception as e:
        logger.error(f"    ❌ ModelScope download failed: {e}")
        return None

def get_dir_size(path):
    """Calculate directory size in MB"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path) * 1024 * 1024 # Recursive returns MB
    except Exception:
        pass
    return total / (1024 * 1024)

# ==============================================================================
# Main Execution
# ==============================================================================
def main(output_dir, selected=None, force=False):
    logger.info("="*60)
    logger.info("🚀 Tianshu Offline Model Downloader")
    logger.info("="*60)
    
    root_path = Path(output_dir).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Output Directory: {root_path}")

    # Filter models
    targets = MODELS
    if selected:
        keys = [s.strip() for s in selected.split(",")]
        targets = {k: v for k, v in MODELS.items() if k in keys}
        logger.info(f"📋 Selected: {', '.join(targets.keys())}")
    else:
        logger.info(f"📋 Downloading ALL {len(targets)} models")

    logger.info("")
    
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    for key, config in targets.items():
        logger.info(f"📦 [{key.upper()}] {config['name']}")
        dest_dir = root_path / config['target_dir']
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Check existing
        if not force and any(dest_dir.iterdir()):
            logger.info(f"    ✅ Found existing files in {dest_dir.name}")
            stats["skipped"] += 1
            logger.info("")
            continue

        success = False
        try:
            if config['source'] == 'huggingface':
                res = download_from_huggingface(
                    config['repo_id'], 
                    dest_dir, 
                    config.get('filename')
                )
                success = bool(res)
            
            elif config['source'] == 'modelscope':
                # ModelScope manages its own cache dir structure slightly differently
                # We point it to the parent of the target dir usually
                res = download_from_modelscope(config['model_id'], dest_dir)
                success = bool(res)
            
            elif config['source'] == 'direct':
                for url in config['urls']:
                    res = download_file_direct(url, dest_dir)
                    if res: success = True
            
            if success:
                stats["success"] += 1
                logger.info(f"    ✅ Download complete")
            else:
                stats["failed"] += 1
                
        except Exception as e:
            logger.error(f"    ❌ Critical error: {e}")
            stats["failed"] += 1
            
        logger.info("")

    # Summary
    logger.info("="*60)
    logger.info(f"📊 Summary: {stats['success']} downloaded, {stats['skipped']} skipped, {stats['failed']} failed")
    logger.info(f"💾 Models are ready at: {root_path}")
    logger.info("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tianshu Model Downloader")
    parser.add_argument("--output", default="./models", help="Target directory (default: ./models)")
    parser.add_argument("--models", help="Specific models to download (e.g., mineru,sensevoice)")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    # Ensure dependencies
    try:
        import huggingface_hub
        import modelscope
        import requests
        import loguru
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}")
        print("Run: pip install huggingface-hub modelscope requests loguru")
        sys.exit(1)

    main(args.output, args.models, args.force)

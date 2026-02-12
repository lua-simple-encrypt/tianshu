#!/usr/bin/env python3
"""
MinerU Tianshu - 统一服务启动器 (Production Version)
天枢 - 企业级 AI 数据预处理平台

启动顺序：
1. API Server (8000) - 任务入口
2. LitServe Worker Pool (8001) - 核心推理引擎
3. Task Scheduler - 队列监控与自动运维
4. MCP Server (8002) - AI 助手扩展接口
"""

import subprocess
import signal
import sys
import time
import os
import json
import argparse
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# 确保能导入 utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import parse_list_arg

class TianshuLauncher:
    """天枢服务集群启动管理器"""

    def __init__(self, args):
        self.args = args
        self.processes = []
        self.output_dir = str(Path(args.output_dir).resolve())
        self.running = True

    def check_environment(self):
        """启动前的环境与模型路径检查"""
        logger.info("🔍 Checking environment...")
        
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 检查本地模型挂载 (D 盘映射)
        if os.getenv("MINERU_MODEL_SOURCE") == "local":
            logger.success("✅ Running in LOCAL model mode (D: drive mounted)")
        
        return True

    def start_services(self):
        """按序启动服务集群"""
        logger.info("=" * 70)
        logger.info("🚀 Starting Tianshu AI Platform Service Cluster")
        logger.info("=" * 70)

        try:
            # 1. 启动 API Server
            logger.info("📡 [1/4] Starting API Server (Port: {})...", self.args.api_port)
            api_env = os.environ.copy()
            api_env["API_PORT"] = str(self.args.api_port)
            api_env["OUTPUT_PATH"] = self.output_dir
            
            api_proc = subprocess.Popen(
                [sys.executable, "api_server.py"],
                cwd=Path(__file__).parent,
                env=api_env
            )
            self.processes.append(("API Server", api_proc))
            time.sleep(2)

            # 2. 启动 LitServe Worker Pool
            logger.info("⚙️  [2/4] Starting LitServe Worker Pool (Port: {})...", self.args.worker_port)
            worker_cmd = [
                sys.executable, "litserve_worker.py",
                "--port", str(self.args.worker_port),
                "--output-dir", self.output_dir,
                "--accelerator", self.args.accelerator,
                "--devices", str(self.args.devices),
                "--workers-per-device", str(self.args.workers_per_device)
            ]

            # 针对 PaddleOCR VLLM 引擎的特殊参数传递
            if self.args.paddleocr_vl_vllm_engine_enabled:
                worker_cmd.append("--paddleocr-vl-vllm-engine-enabled")
                if self.args.paddleocr_vl_vllm_api_list:
                    # 将列表转换为 JSON 字符串传递
                    worker_cmd.extend(["--paddleocr-vl-vllm-api-list", json.dumps(self.args.paddleocr_vl_vllm_api_list)])

            worker_proc = subprocess.Popen(worker_cmd, cwd=Path(__file__).parent)
            self.processes.append(("LitServe Workers", worker_proc))
            time.sleep(5)

            # 3. 启动 Task Scheduler (自动运维)
            logger.info("🔄 [3/4] Starting Task Scheduler...")
            scheduler_cmd = [
                sys.executable, "task_scheduler.py",
                "--litserve-url", f"http://localhost:{self.args.worker_port}/predict",
                "--wait-for-workers"
            ]
            scheduler_proc = subprocess.Popen(scheduler_cmd, cwd=Path(__file__).parent)
            self.processes.append(("Task Scheduler", scheduler_proc))

            # 4. 启动 MCP Server (可选)
            if self.args.enable_mcp:
                logger.info("🔌 [4/4] Starting MCP Server (Port: {})...", self.args.mcp_port)
                mcp_env = os.environ.copy()
                mcp_env["API_BASE_URL"] = f"http://localhost:{self.args.api_port}"
                mcp_env["MCP_PORT"] = str(self.args.mcp_port)
                
                mcp_proc = subprocess.Popen(
                    [sys.executable, "mcp_server.py"],
                    cwd=Path(__file__).parent,
                    env=mcp_env
                )
                self.processes.append(("MCP Server", mcp_proc))

            logger.info("=" * 70)
            logger.success("✅ All Services Online!")
            logger.info("📖 API Dashboard: http://localhost:{}/docs", self.args.api_port)
            return True

        except Exception as e:
            logger.error("❌ Failed to start cluster: {}", e)
            self.stop_services()
            return False

    def stop_services(self, signum=None, frame=None):
        """优雅关闭所有后台进程"""
        logger.info("\n⏹️  Stopping Tianshu Services...")
        for name, proc in reversed(self.processes):
            if proc.poll() is None:
                logger.info("   Stopping {} (PID: {})...", name, proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    logger.info("   ✅ {} stopped", name)
                except:
                    proc.kill()
        
        self.running = False
        sys.exit(0)

    def monitor(self):
        """持续监控进程存活状态"""
        try:
            while self.running:
                for name, proc in self.processes:
                    if proc.poll() is not None:
                        logger.error("❌ Critical Service [{}] unexpectedly stopped!", name)
                        self.stop_services()
                time.sleep(2)
        except KeyboardInterrupt:
            self.stop_services()

def main():
    # 加载环境变量
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    
    parser = argparse.ArgumentParser(description="Tianshu Platform All-in-One Launcher")
    
    # 基础路径与端口
    parser.add_argument("--output-dir", type=str, default="/app/data/output")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--worker-port", type=int, default=8001)
    
    # 推理资源配置
    parser.add_argument("--accelerator", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--devices", type=str, default="auto", help="e.g. '0,1'")
    
    # 扩展引擎配置
    parser.add_argument("--enable-mcp", action="store_true")
    parser.add_argument("--mcp-port", type=int, default=8002)
    parser.add_argument("--paddleocr-vl-vllm-engine-enabled", action="store_true")
    parser.add_argument("--paddleocr-vl-vllm-api-list", type=parse_list_arg, default=[])

    args = parser.parse_args()

    launcher = TianshuLauncher(args)
    
    # 注册系统信号
    signal.signal(signal.SIGINT, launcher.stop_services)
    signal.signal(signal.SIGTERM, launcher.stop_services)

    if launcher.check_environment() and launcher.start_services():
        launcher.monitor()

if __name__ == "__main__":
    main()

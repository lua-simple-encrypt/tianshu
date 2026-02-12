"""
MinerU Tianshu - SQLite Task Database Manager (Production Ready)
天枢任务数据库管理器 - 生产增强版

负责任务的持久化存储、状态管理、原子性操作以及主子任务调度逻辑。
"""

import sqlite3
import json
import uuid
import os
import time
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path
from loguru import logger

# 导入 Redis 队列支持（可选）
try:
    from redis_queue import get_redis_queue
    REDIS_QUEUE_AVAILABLE = True
except ImportError:
    REDIS_QUEUE_AVAILABLE = False
    def get_redis_queue(): return None

class TaskDB:
    """任务数据库管理类：支持混合队列架构 (SQLite + Redis)"""

    def __init__(self, db_path: str = None):
        # 优先级：参数传递 > 环境变量 > 默认路径
        if db_path is None:
            project_root = Path(__file__).parent.parent
            default_db = project_root / "data" / "db" / "mineru_tianshu.db"
            db_path = os.getenv("DATABASE_PATH", str(default_db))
        
        # 确保父目录存在并转换为绝对路径
        db_file = Path(db_path).resolve()
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_file)
        
        self._init_db()

    def _get_conn(self):
        """获取数据库连接：每次新建连接以保证进程/线程安全"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def get_cursor(self):
        """上下文管理器：自动提交和回滚"""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表结构及索引"""
        with self.get_cursor() as cursor:
            # 1. 基础任务表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 0,
                    backend TEXT DEFAULT 'pipeline',
                    options TEXT,
                    result_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    worker_id TEXT,
                    retry_count INTEGER DEFAULT 0,
                    parent_task_id TEXT,
                    is_parent INTEGER DEFAULT 0,
                    child_count INTEGER DEFAULT 0,
                    child_completed INTEGER DEFAULT 0,
                    user_id TEXT
                )
            """)

            # 2. 性能索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_parent_task ON tasks(parent_task_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_parent ON tasks(is_parent)")

    # ============================================================================
    # 核心任务 CRUD
    # ============================================================================

    def create_task(self, file_name: str, file_path: str, backend: str = "pipeline", 
                    options: dict = None, priority: int = 0, user_id: str = None) -> str:
        """创建基础任务并入队"""
        task_id = str(uuid.uuid4())
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO tasks (task_id, file_name, file_path, backend, options, priority, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (task_id, file_name, file_path, backend, json.dumps(options or {}), priority, user_id))

        self._enqueue_to_redis(task_id, priority, {"file_name": file_name, "backend": backend})
        return task_id

    def get_next_task(self, worker_id: str, max_retries: int = 3) -> Optional[Dict]:
        """原子化获取下一个待处理任务（核心防冲突逻辑）"""
        # 1. 优先尝试从 Redis 获取
        task_from_redis = self._get_next_task_redis(worker_id)
        if task_from_redis: return task_from_redis

        # 2. 回退到 SQLite 原子锁模式
        for attempt in range(max_retries):
            try:
                with self.get_cursor() as cursor:
                    cursor.execute("BEGIN IMMEDIATE") # 强制获取排他锁
                    cursor.execute("""
                        SELECT * FROM tasks 
                        WHERE status = 'pending' AND is_parent = 0
                        ORDER BY priority DESC, created_at ASC LIMIT 1
                    """)
                    row = cursor.fetchone()
                    if not row: return None
                    
                    task_id = row["task_id"]
                    cursor.execute("""
                        UPDATE tasks SET status = 'processing', started_at = CURRENT_TIMESTAMP, worker_id = ?
                        WHERE task_id = ? AND status = 'pending'
                    """, (worker_id, task_id))
                    
                    if cursor.rowcount > 0:
                        return dict(row)
            except sqlite3.OperationalError:
                time.sleep(0.1) # 锁竞争，稍后重试
        return None

    def update_task_status(self, task_id: str, status: str, result_path: str = None, 
                           error_message: str = None, worker_id: str = None) -> bool:
        """更新任务状态及结果信息"""
        with self.get_cursor() as cursor:
            if status == "completed":
                sql = "UPDATE tasks SET status=?, completed_at=CURRENT_TIMESTAMP, result_path=? WHERE task_id=?"
                params = (status, result_path, task_id)
            elif status == "failed":
                sql = "UPDATE tasks SET status=?, completed_at=CURRENT_TIMESTAMP, error_message=? WHERE task_id=?"
                params = (status, error_message, task_id)
            else:
                sql = "UPDATE tasks SET status=? WHERE task_id=?"
                params = (status, task_id)
            
            cursor.execute(sql, params)
            success = cursor.rowcount > 0
            
            if success and status in ["completed", "failed"]:
                self._notify_redis_task_done(task_id, worker_id or "", status)
            return success

    # ============================================================================
    # 主子任务支持 (大文件拆分关键逻辑)
    # ============================================================================

    def convert_to_parent_task(self, task_id: str, child_count: int = 0):
        """将当前任务标记为父任务并暂停处理"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE tasks 
                SET is_parent = 1, child_count = ?, status = 'processing' 
                WHERE task_id = ?
            """, (child_count, task_id))
        logger.info(f"🔄 Task {task_id} converted to parent (expecting {child_count} children)")

    def create_child_task(self, parent_task_id: str, file_name: str, file_path: str, 
                          backend: str, options: dict, priority: int = 0, user_id: str = None) -> str:
        """创建子分片任务"""
        task_id = str(uuid.uuid4())
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO tasks (
                    task_id, parent_task_id, file_name, file_path, backend, 
                    options, status, priority, user_id, is_parent
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, 0)
            """, (task_id, parent_task_id, file_name, file_path, backend, json.dumps(options), priority, user_id))
        
        self._enqueue_to_redis(task_id, priority, {"file_name": file_name, "is_child": True})
        return task_id

    def on_child_task_completed(self, child_task_id: str) -> Optional[str]:
        """子任务完成回调：增加父任务计数，全完成则触发合并"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT parent_task_id FROM tasks WHERE task_id = ?", (child_task_id,))
            res = cursor.fetchone()
            if not res or not res["parent_task_id"]: return None
            
            parent_id = res["parent_task_id"]
            cursor.execute("UPDATE tasks SET child_completed = child_completed + 1 WHERE task_id = ?", (parent_id,))
            
            # 检查计数
            cursor.execute("SELECT child_count, child_completed FROM tasks WHERE task_id = ?", (parent_id,))
            p = cursor.fetchone()
            if p and p["child_completed"] >= p["child_count"]:
                return parent_id # 返回父 ID 告知 Worker 该合并了
        return None

    def on_child_task_failed(self, child_task_id: str, error_message: str):
        """子任务失败逻辑：连锁标记父任务失败"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT parent_task_id FROM tasks WHERE task_id = ?", (child_task_id,))
            row = cursor.fetchone()
            if row and row["parent_task_id"]:
                parent_id = row["parent_task_id"]
                cursor.execute("""
                    UPDATE tasks SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error_message = ?
                    WHERE task_id = ?
                """, (f"Child failure ({child_task_id}): {error_message}", parent_id))

    # ============================================================================
    # 辅助工具与维护
    # ============================================================================

    def get_task(self, task_id: str) -> Optional[Dict]:
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_child_tasks(self, parent_task_id: str) -> List[Dict]:
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM tasks WHERE parent_task_id = ?", (parent_task_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_queue_stats(self) -> Dict[str, int]:
        with self.get_cursor() as cursor:
            cursor.execute("SELECT status, COUNT(*) as count FROM tasks GROUP BY status")
            return {row["status"]: row["count"] for row in cursor.fetchall()}

    def cleanup_old_task_records(self, days: int = 7):
        """物理删除过期文件及记录"""
        import shutil
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT task_id, file_path, result_path FROM tasks 
                WHERE completed_at < datetime('now', '-' || ? || ' days')
                AND status IN ('completed', 'failed')
            """, (days,))
            for row in cursor.fetchall():
                # 删除物理文件
                for path in [row["file_path"], row["result_path"]]:
                    if path and Path(path).exists():
                        if Path(path).is_file(): Path(path).unlink()
                        else: shutil.rmtree(path, ignore_errors=True)
            
            cursor.execute("DELETE FROM tasks WHERE completed_at < datetime('now', '-' || ? || ' days')", (days,))
            return cursor.rowcount

    def reset_stale_tasks(self, timeout_minutes: int = 60):
        """将长时间卡在 processing 的任务重置为 pending"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE tasks SET status = 'pending', worker_id = NULL, retry_count = retry_count + 1
                WHERE status = 'processing' AND started_at < datetime('now', '-' || ? || ' minutes')
            """, (timeout_minutes,))
            return cursor.rowcount

    # ============================================================================
    # Redis 内部逻辑
    # ============================================================================
    def _enqueue_to_redis(self, task_id, priority, data):
        if REDIS_QUEUE_AVAILABLE:
            q = get_redis_queue()
            if q: q.enqueue(task_id, priority, data)

    def _get_next_task_redis(self, worker_id):
        if not REDIS_QUEUE_AVAILABLE: return None
        q = get_redis_queue()
        if not q: return None
        tid = q.dequeue(worker_id)
        if tid: return self.get_task(tid)
        return None

    def _notify_redis_task_done(self, tid, wid, status):
        if REDIS_QUEUE_AVAILABLE:
            q = get_redis_queue()
            if q:
                if status == "completed": q.complete(tid, wid)
                else: q.fail(tid, wid)

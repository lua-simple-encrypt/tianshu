from typing import Any
import httpx
import logging
import os

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

# 禁用代理，避免 Squid 拦截本地请求
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 如果没有 handler，添加一个控制台 handler
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - [Tianshu Plugin] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TianshuProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """
        验证 API 配置
        直接使用 API Key 验证接口有效性
        """
        logger.info("=" * 80)
        logger.info("开始验证 Tianshu API 凭据")
        logger.info("=" * 80)

        try:
            api_base_url = credentials.get("api_base_url", "").rstrip("/")
            api_key = credentials.get("api_key", "")

            logger.info(f"📡 API Base URL: {api_base_url}")
            logger.info(f"🔑 API Key 长度: {len(api_key) if api_key else 0} 字符")

            if len(api_key) > 20:
                masked_key = f"{api_key[:8]}...{api_key[-8:]}"
                logger.debug(f"🔑 API Key (部分): {masked_key}")

            if not api_base_url:
                logger.error("❌ API Base URL 为空")
                raise ValueError("API Base URL is required")
            if not api_key:
                logger.error("❌ API Key 为空")
                raise ValueError("API Key is required")

            # 使用 API Key 调用队列接口验证
            test_url = f"{api_base_url}/api/v1/queue/tasks?limit=1"
            headers = {
                "X-API-Key": api_key,
                "Accept": "application/json",
                "User-Agent": "Dify-Plugin-Tianshu/1.0"
            }

            logger.info("📤 发送验证请求:")
            logger.info(f"   URL: {test_url}")
            logger.info("   Method: GET")
            logger.info(f"   Headers: {headers}")

            try:
                # 直接发送请求（已通过环境变量禁用代理）
                response = httpx.get(test_url, headers=headers, timeout=10.0)

                logger.info("📥 收到响应:")
                logger.info(f"   状态码: {response.status_code}")
                logger.info(f"   响应头: {dict(response.headers)}")
                logger.info(f"   请求的实际 URL: {response.url}")

                # 记录响应内容（截取前 1000 字符）
                response_text = response.text
                if len(response_text) > 1000:
                    logger.info(f"   响应内容 (前1000字符): {response_text[:1000]}...")
                else:
                    logger.info(f"   响应内容: {response_text}")

                if response.status_code == 401:
                    logger.error("❌ 认证失败 (401)")
                    logger.error("   原因: API Key 无效或格式错误")
                    logger.error(f"   详细信息: {response_text}")
                    raise ValueError(f"Invalid API Key: Authentication failed (401). Detail: {response_text}")

                elif response.status_code == 403:
                    logger.error("❌ 权限不足 (403)")
                    logger.error("   可能原因:")
                    logger.error("   1. 用户未激活 (is_active=False)")
                    logger.error("   2. 缺少必要权限")
                    logger.error(f"   服务器响应: {response_text}")
                    raise ValueError(f"API Key validation failed (403 Forbidden). Server response: {response_text}")

                elif response.status_code == 200:
                    logger.info("✅ 验证成功!")
                    logger.info("   API Key 有效且用户已激活")

                    # 尝试解析响应 JSON
                    try:
                        json_data = response.json()
                        if "tasks" in json_data:
                            logger.info(f"   返回任务数: {len(json_data.get('tasks', []))}")
                        if "can_view_all" in json_data:
                            logger.info(f"   全局查看权限: {json_data.get('can_view_all')}")
                    except Exception:
                        pass

                    logger.info("=" * 80)
                    return

                else:
                    logger.error(f"❌ 验证失败 - 未预期的状态码: {response.status_code}")
                    logger.error(f"   响应内容: {response_text}")
                    raise ValueError(f"API validation failed with status {response.status_code}: {response_text}")

            except httpx.TimeoutException as e:
                logger.error(f"❌ 请求超时: {str(e)}")
                raise ValueError(f"Request timeout: {str(e)}")

            except httpx.ConnectError as e:
                logger.error(f"❌ 连接失败: {str(e)}")
                logger.error("   请检查:")
                logger.error("   1. Tianshu 服务是否正在运行")
                logger.error(f"   2. URL 是否正确: {api_base_url}")
                logger.error("   3. 网络连接是否正常")
                raise ValueError(f"Cannot connect to API server: {str(e)}")

            except httpx.RequestError as e:
                logger.error(f"❌ 网络请求错误: {str(e)}")
                raise ValueError(f"Request error: {str(e)}")

        except ValueError as e:
            logger.error(f"❌ 凭据验证失败: {str(e)}")
            logger.info("=" * 80)
            raise ToolProviderCredentialValidationError(str(e))

        except Exception as e:
            logger.exception("❌ 未预期的错误:")
            logger.info("=" * 80)
            raise ToolProviderCredentialValidationError(f"Unexpected error: {str(e)}")

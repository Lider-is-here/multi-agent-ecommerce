"""编排器智能体 — 基于 LangGraph 的实现。

本模块提供基于 LangGraph 的编排器智能体，是整个多智能体系统的入口点。
负责：
1. 接收用户请求
2. 分析意图并选择合适的专业智能体
3. 通过 A2A 协议调用专业智能体
4. 将结果返回给用户
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

import httpx
from pydantic import Field

from orchestrator.prompts import SYSTEM_PROMPT
from shared.config import settings
from shared.context import (
    current_session_id,
    current_steps,
    current_stream_queue,
    current_user_email,
    current_user_role,
)
from shared.langgraph_agent import (
    LangGraphAgentConfig,
    create_langgraph_agent,
    run_agent,
)
from shared.oauth.service_client import build_a2a_headers
from shared.telemetry import a2a_call_span

logger = logging.getLogger(__name__)

AGENT_REGISTRY: dict[str, str] = json.loads(settings.AGENT_REGISTRY)


async def call_specialist_agent(
    agent_name: Annotated[str, Field(description="要调用的专业智能体名称")],
    message: Annotated[str, Field(description="要发送给专业智能体的消息/请求")],
) -> str:
    """通过 A2A 协议将请求路由到专业智能体。

    可用智能体: product-discovery, order-management, pricing-promotions,
    inventory-fulfillment
    """
    url = AGENT_REGISTRY.get(agent_name)
    if not url:
        available = ", ".join(AGENT_REGISTRY.keys()) if AGENT_REGISTRY else "none configured"
        return f"未知智能体: {agent_name}。可用智能体: {available}"

    logger.info("a2a.call source=orchestrator target=%s user=%s", agent_name, current_user_email.get())

    stream_queue = current_stream_queue.get()
    headers = await build_a2a_headers(
        user_email=current_user_email.get(),
        user_role=current_user_role.get() or "customer",
        session_id=current_session_id.get(),
    )
    request_body = {"message": message}

    with a2a_call_span("orchestrator", agent_name, url):
        if stream_queue is not None:
            try:
                chunks: list[str] = []
                current_event: str = "data"
                async with httpx.AsyncClient(timeout=60) as client:
                    async with client.stream(
                        "POST",
                        f"{url}/message:stream",
                        json=request_body,
                        headers=headers,
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line.startswith("event: "):
                                current_event = line[7:].strip()
                                continue
                            if not line:
                                current_event = "data"
                                continue
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:]

                            if current_event == "step":
                                try:
                                    step_data = json.loads(payload)
                                    bucket = current_steps.get()
                                    if bucket is not None:
                                        bucket.append(step_data)
                                except (json.JSONDecodeError, ValueError):
                                    pass
                                current_event = "data"
                                continue

                            if payload == "[DONE]":
                                break
                            if payload.startswith("[ERROR"):
                                logger.error("a2a.stream_error target=%s payload=%s", agent_name, payload)
                                continue

                            chunks.append(payload)
                            await stream_queue.put(("delta", agent_name, payload))

                return "".join(chunks) or f"{agent_name} 智能体返回了空响应。"

            except (httpx.TimeoutException, httpx.HTTPStatusError, Exception) as exc:
                logger.warning("a2a.stream_fallback target=%s reason=%s", agent_name, type(exc).__name__)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{url}/message:send",
                    json=request_body,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

                bucket = current_steps.get()
                specialist_steps = data.get("steps") or []
                if bucket is not None and specialist_steps:
                    bucket.extend(specialist_steps)

                return data.get("response", resp.text)

        except httpx.TimeoutException:
            logger.error("a2a.timeout target=%s", agent_name)
            return f"{agent_name} 智能体响应超时。请重试。"
        except httpx.HTTPStatusError as e:
            logger.error("a2a.error target=%s status=%s", agent_name, e.response.status_code)
            return f"{agent_name} 智能体返回错误（状态码 {e.response.status_code}）。请重试。"
        except Exception:
            logger.exception("a2a.failure target=%s", agent_name)
            return f"无法连接到 {agent_name} 智能体。请稍后重试。"


ORCHESTRATOR_TOOLS = [call_specialist_agent]


def create_orchestrator_agent() -> Any:
    """创建基于 LangGraph 的编排器智能体。"""
    config = LangGraphAgentConfig(
        name="orchestrator",
        instructions=SYSTEM_PROMPT,
        tools=ORCHESTRATOR_TOOLS,
        temperature=0.3,
        max_iterations=20,
    )
    return create_langgraph_agent(config)


async def run_orchestrator_agent(
    user_message: str,
    metadata: dict[str, Any] | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """运行编排器智能体。"""
    agent = create_orchestrator_agent()
    return await run_agent(agent, user_message, metadata, message_history)

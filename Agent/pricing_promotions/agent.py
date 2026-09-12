"""定价与促销智能体 — 基于 LangGraph 的实现。"""

from __future__ import annotations

from typing import Any

from pricing_promotions.prompts import SYSTEM_PROMPT
from pricing_promotions.tools import get_active_deals
from shared.langgraph_agent import (
    LangGraphAgentConfig,
    create_langgraph_agent,
    run_agent,
)
from shared.tools.pricing_tools import get_price_history
from shared.tools.user_tools import get_purchase_history, get_user_profile

AGENT_TOOLS = [
    get_active_deals,
    get_price_history,
    get_user_profile,
    get_purchase_history,
]


def create_pricing_promotions_agent() -> Any:
    """创建基于 LangGraph 的定价与促销智能体。"""
    config = LangGraphAgentConfig(
        name="pricing-promotions",
        instructions=SYSTEM_PROMPT,
        tools=AGENT_TOOLS,
        temperature=0.7,
        max_iterations=15,
    )
    return create_langgraph_agent(config)


async def run_pricing_promotions_agent(
    user_message: str,
    metadata: dict[str, Any] | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """运行定价与促销智能体。"""
    agent = create_pricing_promotions_agent()
    return await run_agent(agent, user_message, metadata, message_history)

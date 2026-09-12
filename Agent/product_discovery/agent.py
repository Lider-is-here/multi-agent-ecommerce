"""产品发现智能体 — 基于 LangGraph 的实现。

本模块提供基于 LangGraph 的产品发现智能体，负责：
- 自然语言产品搜索与个性化推荐
- 产品详情查询、对比、趋势分析
- 语义搜索、库存检查、价格历史
"""

from __future__ import annotations

import logging
from typing import Any

from product_discovery.prompts import SYSTEM_PROMPT
from product_discovery.tools import (
    compare_products,
    find_similar_products,
    get_product_details,
    get_trending_products,
    search_products,
    semantic_search,
)
from shared.langgraph_agent import (
    LangGraphAgentConfig,
    create_langgraph_agent,
    run_agent,
)
from shared.tools.inventory_tools import check_stock, get_warehouse_availability
from shared.tools.memory_tools import recall_memories, store_memory
from shared.tools.pricing_tools import get_price_history
from shared.tools.user_tools import get_purchase_history, get_user_profile

logger = logging.getLogger(__name__)

AGENT_TOOLS = [
    search_products,
    get_product_details,
    compare_products,
    semantic_search,
    find_similar_products,
    get_trending_products,
    check_stock,
    get_warehouse_availability,
    get_price_history,
    get_user_profile,
    get_purchase_history,
    store_memory,
    recall_memories,
]


def create_product_discovery_agent() -> Any:
    """创建基于 LangGraph 的产品发现智能体。"""
    config = LangGraphAgentConfig(
        name="product-discovery",
        instructions=SYSTEM_PROMPT,
        tools=AGENT_TOOLS,
        temperature=0.7,
        max_iterations=15,
    )
    return create_langgraph_agent(config)


async def run_product_discovery_agent(
    user_message: str,
    metadata: dict[str, Any] | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """运行产品发现智能体。"""
    agent = create_product_discovery_agent()
    return await run_agent(agent, user_message, metadata, message_history)

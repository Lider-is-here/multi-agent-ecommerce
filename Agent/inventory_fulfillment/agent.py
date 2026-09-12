"""库存与履行智能体 —— 基于 LangGraph 的实现。

本模块提供基于 LangGraph 的库存与履行智能体，负责：
- 库存查询与补货计划
- 运输估价与承运商比较
- 订单追踪与履行计划
- 缺货订单管理
"""

from __future__ import annotations

import logging
from typing import Any

from inventory_fulfillment.prompts import SYSTEM_PROMPT
from inventory_fulfillment.tools import (
    calculate_fulfillment_plan,
    compare_carriers,
    estimate_shipping,
    get_restock_schedule,
    get_tracking_status,
    place_backorder,
)
from shared.config import settings
from shared.langgraph_agent import (
    LangGraphAgentConfig,
    create_langgraph_agent,
    run_agent,
)
from shared.tools.inventory_tools import check_stock, get_warehouse_availability
from shared.tools.user_tools import get_user_profile

logger = logging.getLogger(__name__)

AGENT_TOOLS = [
    check_stock,
    get_warehouse_availability,
    get_restock_schedule,
    estimate_shipping,
    compare_carriers,
    get_tracking_status,
    calculate_fulfillment_plan,
    place_backorder,
    get_user_profile,
]


def create_inventory_fulfillment_agent() -> Any:
    """创建基于 LangGraph 的库存与履行智能体。"""
    config = LangGraphAgentConfig(
        name="inventory-fulfillment",
        instructions=SYSTEM_PROMPT,
        tools=AGENT_TOOLS,
        temperature=0.2,
        max_iterations=15,
    )
    return create_langgraph_agent(config)


async def refresh_mcp_auth() -> None:
    """MCP 认证刷新（保留接口，当前为空闲操作）。

    MCP 功能在本简化版本中未启用；当 ``settings.MCP_ENABLED`` 为 True 且
    ``settings.MCP_AUTH_ENABLED`` 为 True 时，此函数可扩展为获取服务令牌。
    """
    if not (settings.MCP_ENABLED and settings.MCP_AUTH_ENABLED):
        return
    logger.info("MCP auth refresh requested (no-op in simplified LangGraph mode)")


async def run_inventory_fulfillment_agent(
    user_message: str,
    metadata: dict[str, Any] | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """运行库存与履行智能体。"""
    agent = create_inventory_fulfillment_agent()
    return await run_agent(agent, user_message, metadata, message_history)

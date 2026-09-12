"""订单管理智能体 — 基于 LangGraph 的实现。

本模块提供基于 LangGraph 的订单管理智能体，负责：
- 订单查询、追踪、取消、修改
- 退货、退款处理
- 购物车管理、地址管理
"""

from __future__ import annotations

import logging
from typing import Any

from order_management.prompts import SYSTEM_PROMPT
from order_management.tools import (
    cancel_order,
    get_order_details,
    get_order_tracking,
    get_user_orders,
    modify_order,
)
from shared.langgraph_agent import (
    LangGraphAgentConfig,
    create_langgraph_agent,
    run_agent,
)
from shared.tools.cart_tools import (
    add_to_cart,
    get_cart,
    remove_from_cart,
    set_billing_address,
    set_billing_same_as_shipping,
    set_shipping_address,
    update_cart_quantity,
)
from shared.tools.return_tools import (
    check_return_eligibility,
    get_return_status,
    initiate_return,
    process_refund,
)
from shared.tools.user_tools import get_user_profile

logger = logging.getLogger(__name__)

AGENT_TOOLS = [
    get_user_orders,
    get_order_details,
    get_order_tracking,
    cancel_order,
    modify_order,
    check_return_eligibility,
    initiate_return,
    process_refund,
    get_return_status,
    get_user_profile,
    add_to_cart,
    get_cart,
    remove_from_cart,
    update_cart_quantity,
    set_shipping_address,
    set_billing_address,
    set_billing_same_as_shipping,
]


def create_order_management_agent() -> Any:
    """创建基于 LangGraph 的订单管理智能体。"""
    config = LangGraphAgentConfig(
        name="order-management",
        instructions=SYSTEM_PROMPT,
        tools=AGENT_TOOLS,
        temperature=0.2,
        max_iterations=15,
    )
    return create_langgraph_agent(config)


async def run_order_management_agent(
    user_message: str,
    metadata: dict[str, Any] | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """运行订单管理智能体。"""
    agent = create_order_management_agent()
    return await run_agent(agent, user_message, metadata, message_history)

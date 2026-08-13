"""
chatbot.py
==========
A multi-turn customer-support chatbot with tool calling, built with Claude.

Tools available (same data as agent_plain.py):
  - get_order_status(order_id)   → real-time shipping status from in-memory DB
  - get_refund_policy(category)  → refund rules per product category

Unlike agent_plain.py (which delegates the tool-call loop to LangGraph),
this chatbot manages the loop manually so the full message history — including
tool calls and tool results — is preserved across turns. This is what makes
multi-turn evaluation possible with DeepEval.

chat() returns:
  - reply        : the final assistant text shown to the user
  - history      : updated message history to pass into the next turn
  - tools_called : list of (tool_name, args, result) for this turn

Usage:
    history = []
    reply, history, tools = chat("Where is order ORD-1042?", history)
    reply, history, tools = chat("What about the refund policy?", history)

How to run standalone:
    python chatbot.py
"""

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# In-memory data (same as agent_plain.py)
# ---------------------------------------------------------------------------
ORDERS = {
    "ORD-1042": {"status": "Shipped",    "eta": "2026-05-13"},
    "ORD-2099": {"status": "Delivered",  "eta": "2026-05-08"},
    "ORD-7777": {"status": "Processing", "eta": "2026-05-15"},
}

REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 15 days, unopened.",
    "clothing":    "Clothing can be returned within 30 days with tags attached.",
    "food":        "Food items are non-returnable for safety reasons.",
    "furniture":   "Furniture can be returned within 30 days if unassembled.",
}

# ---------------------------------------------------------------------------
# Tool definitions — passed to the Anthropic API
# (Anthropic's tool schema is flatter than OpenAI's: no "type": "function"
#  wrapper, and the JSON schema key is "input_schema" instead of "parameters".)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_order_status",
        "description": "Look up the shipping status of a customer order by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. ORD-1042",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "get_refund_policy",
        "description": "Return the refund/return policy for a product category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Product category, e.g. electronics, clothing, food",
                }
            },
            "required": ["category"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------
def _execute_tool(name: str, args: dict) -> str:
    if name == "get_order_status":
        order_id = args.get("order_id", "").upper()
        order = ORDERS.get(order_id)
        if not order:
            return f"No order found with ID {order_id}."
        return f"Order {order_id} is {order['status']}. ETA: {order['eta']}."

    if name == "get_refund_policy":
        category = args.get("category", "").lower()
        policy = REFUND_POLICIES.get(category)
        if not policy:
            return f"No refund policy on file for '{category}'."
        return policy

    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly and professional customer-support chatbot \
for ShopEasy, an online retail store.

You have access to two tools:
  - get_order_status : use whenever a customer asks about a specific order
  - get_refund_policy: use whenever a customer asks about returns or refunds

Rules:
- Always use the tools when you have the information needed to call them.
- Be concise and polite.
- Never discuss topics outside of ShopEasy customer support.
- Remember everything the customer tells you in the current conversation."""


# ---------------------------------------------------------------------------
# chat() — one user turn, returns reply + updated history + tools used
# ---------------------------------------------------------------------------
def chat(
    user_message: str,
    history: list[dict],
) -> tuple[str, list[dict], list[dict]]:
    """
    Send one user message, handle any tool calls, return the final reply.

    Returns:
        reply        : final assistant text
        history      : updated message list (include in next call)
        tools_called : list of {"name": ..., "args": ..., "result": ...}
                       for all tools the model called this turn
    """
    history = history + [{"role": "user", "content": user_message}]
    tools_called = []

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=history,
            tools=TOOLS,
            temperature=0,
        )

        # No tool call — final answer.
        # Anthropic signals this via stop_reason, not an empty tool_calls list.
        if response.stop_reason != "tool_use":
            # response.content is a list of blocks; grab the text block(s).
            reply = "".join(
                block.text for block in response.content if block.type == "text"
            )
            history = history + [{"role": "assistant", "content": response.content}]
            return reply, history, tools_called

        # Model wants to call one or more tools.
        # Append the assistant's full content (text + tool_use blocks) to history.
        history = history + [{"role": "assistant", "content": response.content}]

        # Anthropic expects ALL tool results for this turn bundled into a
        # single "user" message, as a list of tool_result blocks.
        tool_result_blocks = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            args = block.input
            result = _execute_tool(block.name, args)
            tools_called.append({
                "name":   block.name,
                "args":   args,
                "result": result,
            })
            tool_result_blocks.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result,
            })

        history = history + [{"role": "user", "content": tool_result_blocks}]
        # Loop back so the model can respond to the tool result(s).


# ---------------------------------------------------------------------------
# Standalone interactive demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("ShopEasy Support Chatbot (with tools) — type 'quit' to exit\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        reply, history, tools = chat(user_input, history)
        if tools:
            for t in tools:
                print(f"  [tool] {t['name']}({t['args']}) → {t['result']}")
        print(f"Bot: {reply}\n")
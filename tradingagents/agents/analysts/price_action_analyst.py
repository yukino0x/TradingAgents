from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_stock_data,
    get_indicators,
)


def create_price_action_analyst(llm):
    def price_action_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are an elite Price Action Analyst focused on discretionary chart reading and market micro-structure behavior.

Your mission:
1) Use pure price behavior first (candlestick structure, market structure, swing points, momentum shifts, volatility regime).
2) Use indicators only as secondary confirmation, never as the primary thesis.
3) Deliver a trading-grade report with explicit risk framing, invalidation levels, and scenario planning.

Mandatory workflow:
- First, call get_stock_data(symbol, start_date, end_date) to retrieve OHLCV data.
- Then, if needed, call get_indicators(symbol, indicator, curr_date, look_back_days) to validate momentum/volatility context.
- For indicators, prioritize: close_10_ema, close_50_sma, close_200_sma, atr, macd, rsi, boll_ub, boll_lb.

Analysis framework (must follow):
- Market Structure: identify higher highs/lows vs lower highs/lows, trend phase, transitions.
- Candlestick Behavior: rejection candles, expansion bars, inside/outside bars, failed breakouts/breakdowns.
- Key Levels: mark support/resistance, prior day/week highs/lows, breakdown/breakout pivots.
- Volatility & Range: regime classification (expanding, contracting, rotational), ATR-aware stop logic.
- Participation Proxy: use volume in OHLCV to evaluate conviction and exhaustion.
- Trade Scenarios:
  - Bull case: trigger, confirmation, stop, invalidation, target path.
  - Bear case: trigger, confirmation, stop, invalidation, target path.
  - Neutral case: conditions that imply no-trade / wait.

Output requirements:
- Be concrete and evidence-based. Quote specific dates and price zones from tool output.
- Avoid vague statements like “looks bullish”; explain why with structural evidence.
- End with a confidence score (0-100) and top 3 risks that can break your thesis.
- Append a Markdown table summarizing:
  | Scenario | Trigger | Confirmation | Invalidation | Target | Risk Notes |
"""
            + """ Keep recommendations realistic, execution-aware, and tightly tied to retrieved data."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "price_action_report": report,
        }

    return price_action_analyst_node


import numpy as np
import pandas as pd

from portfolio_backtest import (
    STARTING_CAPITAL,
    MAX_POSITIONS,
    COST_PER_SIDE,
    simulate_portfolio,
)


def main():
    # Seven days of synthetic data with one known trade.
    dates = pd.date_range(
        "2025-01-01", periods=7, freq="D"
    )

    entry_open = 100.0
    exit_close = 110.0

    rows = []

    for i, date in enumerate(dates):
        rows.append({
            "date": date,
            "symbol": "TESTUSDT",
            "open": entry_open,
            "close": exit_close if i == 6 else 100.0,
            "quote_volume": 1000.0,
            "score": 4,
            "return_7d": 0.10,
            "volume_ratio": 2.0,
        })

    data = pd.DataFrame(rows)

    # Independently calculate the expected cash flows.
    budget = STARTING_CAPITAL / MAX_POSITIONS

    quantity = budget / (
        entry_open * (1 + COST_PER_SIDE)
    )

    expected_final_equity = (
        STARTING_CAPITAL
        - budget
        + quantity * exit_close * (1 - COST_PER_SIDE)
    )

    expected_return_pct = (
        expected_final_equity / STARTING_CAPITAL - 1
    ) * 100

    expected_net_trade_return_pct = (
        (
            exit_close * (1 - COST_PER_SIDE)
        )
        / (
            entry_open * (1 + COST_PER_SIDE)
        )
        - 1
    ) * 100

    result = simulate_portfolio(
        data=data,
        dates=list(dates),
        mode="momentum",
    )

    print("\nNimiQuant — Synthetic Accounting Test")
    print("=" * 46)

    print(
        f"Expected final equity: "
        f"${expected_final_equity:.6f}"
    )
    print(
        f"Actual final equity:   "
        f"${result['final_equity']:.6f}"
    )

    print(
        f"Expected total return: "
        f"{expected_return_pct:.6f}%"
    )
    print(
        f"Actual total return:   "
        f"{result['total_return_pct']:.6f}%"
    )

    print(
        f"Expected net trade return: "
        f"{expected_net_trade_return_pct:.6f}%"
    )
    print(f"Trades recorded: {result['trades']}")

    expected_win_rate = (
        100.0 if expected_net_trade_return_pct > 0 else 0.0
    )

    print(
        f"Expected win rate: {expected_win_rate:.2f}%"
    )
    print(
        f"Actual win rate:   {result['win_rate_pct']:.2f}%"
    )

    assert result["trades"] == 1, (
        f"Expected 1 trade, got {result['trades']}"
    )

    assert np.isclose(
        result["final_equity"],
        expected_final_equity,
        atol=1e-8,
    ), "Final equity does not match the independent calculation."

    assert np.isclose(
        result["total_return_pct"],
        expected_return_pct,
        atol=1e-8,
    ), "Total return does not match the independent calculation."

    assert np.isclose(
        result["win_rate_pct"],
        expected_win_rate,
        atol=1e-8,
    ), "Win rate does not match the expected result."

    print("\nPASS: Synthetic accounting checks match.")


if __name__ == "__main__":
    main()

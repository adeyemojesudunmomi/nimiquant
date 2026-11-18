
"""
NimiQuant — Breakout Research
Project 1.0 | Built by Nimi

Evaluates forward returns after breakout signals.
Research only; not a trading recommendation.
"""

from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/prices.csv")
SIGNALS_FILE = Path("data/breakout_signals.csv")
OUTPUT_FILE = Path("data/breakout_research_results.csv")

HOLD_DAYS_LIST = [3, 5, 7]

# Assumptions per side: 0.10% fee + 0.05% slippage.
FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005
COST_PER_SIDE = FEE_PER_SIDE + SLIPPAGE_PER_SIDE


def main():
    if not DATA_FILE.exists() or not SIGNALS_FILE.exists():
        print("Missing input file. Run data.py and breakout_model.py first.")
        return

    prices = pd.read_csv(DATA_FILE, parse_dates=["date"])
    signals = pd.read_csv(SIGNALS_FILE, parse_dates=["date"])

    required = {"date", "symbol", "open", "close"}
    if not required.issubset(prices.columns):
        print(f"Price data must contain: {sorted(required)}")
        return

    if not {"date", "symbol"}.issubset(signals.columns):
        print("Breakout signals must contain date and symbol columns.")
        return

    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()

    prices = (
        prices.drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .copy()
    )

    # Entry is the next day's open because the breakout is only
    # confirmed after the signal day's candle closes.
    grouped = prices.groupby("symbol", sort=False)
    prices["entry_open"] = grouped["open"].shift(-1)

    for days in HOLD_DAYS_LIST:
        prices[f"exit_close_{days}d"] = grouped["close"].shift(-(days + 1))

    future_columns = [
        "symbol", "date", "entry_open",
        *[f"exit_close_{days}d" for days in HOLD_DAYS_LIST],
    ]

    evaluated = signals.merge(
        prices[future_columns],
        on=["symbol", "date"],
        how="left",
        validate="many_to_one",
    )

    summary = []

    for days in HOLD_DAYS_LIST:
        exit_column = f"exit_close_{days}d"

        trades = evaluated.dropna(
            subset=["entry_open", exit_column]
        ).copy()

        trades = trades[
            (trades["entry_open"] > 0) & (trades[exit_column] > 0)
        ]

        if trades.empty:
            summary.append({
                "hold_days": days,
                "signals_tested": 0,
                "win_rate_pct": float("nan"),
                "average_gross_return_pct": float("nan"),
                "average_net_return_pct": float("nan"),
                "median_net_return_pct": float("nan"),
            })
            continue

        gross = (
            trades[exit_column] / trades["entry_open"] - 1
        ) * 100

        # Approximate entry and exit costs on the traded notional.
        net = (
            trades[exit_column] * (1 - COST_PER_SIDE)
            / (trades["entry_open"] * (1 + COST_PER_SIDE)) - 1
        ) * 100

        summary.append({
            "hold_days": days,
            "signals_tested": len(trades),
            "win_rate_pct": round((net > 0).mean() * 100, 2),
            "average_gross_return_pct": round(gross.mean(), 3),
            "average_net_return_pct": round(net.mean(), 3),
            "median_net_return_pct": round(net.median(), 3),
        })

    results = pd.DataFrame(summary)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nNimiQuant — Breakout Research Results")
    print("=" * 65)
    print(results.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(
        "\nEstimated costs per side: "
        f"{COST_PER_SIDE * 100:.2f}%"
    )
    print("Repeated and overlapping signals may not be independent.")
    print("This event study does not simulate a complete trading portfolio.")


if __name__ == "__main__":
    main()

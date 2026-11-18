
"""
NimiQuant — Crypto Intelligence Engine
Project 1.0 | Built by Nimi

Historical evaluation of model signals against
random selection from the eligible coin universe.

Research only — not financial advice.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from model import calculate_signals


DATA_FILE = Path("data/prices.csv")
OUTPUT_FILE = Path("data/research_results.csv")

HOLD_DAYS = 7
MIN_HISTORY = 60
TOP_COINS_PER_DAY = 5
RANDOM_TRIALS = 100
RANDOM_SEED = 42


def main():
    if not DATA_FILE.exists():
        print("Price data not found. Run data.py first.")
        return

    required_columns = {"date", "symbol", "open", "close"}

    prices = pd.read_csv(DATA_FILE, parse_dates=["date"])

    if not required_columns.issubset(prices.columns):
        missing = required_columns - set(prices.columns)
        print(f"Price data is missing required columns: {sorted(missing)}")
        return

    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    prices = prices.sort_values(["symbol", "date"]).copy()

    # A signal is generated after the day's close.
    # Entry: next day's open.
    # Exit: close HOLD_DAYS days after entry.
    grouped = prices.groupby("symbol", sort=False)

    prices["entry_open"] = grouped["open"].shift(-1)
    prices["exit_close"] = grouped["close"].shift(-(HOLD_DAYS + 1))

    prices["future_return_pct"] = (
        prices["exit_close"] / prices["entry_open"] - 1
    ) * 100

    historical_signals = []
    eligible_pools = {}

    all_dates = sorted(prices["date"].unique())

    for current_date in all_dates:
        day_prices = prices[prices["date"] == current_date]

        # Only include coins with enough history and a complete
        # future entry/exit outcome.
        history = prices[prices["date"] <= current_date]
        history_counts = history.groupby("symbol").size()

        eligible_symbols = history_counts[
            history_counts >= MIN_HISTORY
        ].index

        eligible = day_prices[
            day_prices["symbol"].isin(eligible_symbols)
        ].dropna(
            subset=["entry_open", "exit_close", "future_return_pct"]
        ).copy()

        eligible = eligible[
            (eligible["entry_open"] > 0)
            & (eligible["exit_close"] > 0)
        ]

        if eligible.empty:
            continue

        # Keep the full eligible universe for a fair random baseline.
        eligible_pools[pd.Timestamp(current_date)] = eligible[
            ["symbol", "future_return_pct"]
        ].copy()

        history = history[
            history["symbol"].isin(eligible_symbols)
        ]

        signals = calculate_signals(history)

        # calculate_signals may return an empty DataFrame without columns.
        if signals.empty or "date" not in signals.columns:
            continue

        signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
        signals = signals[signals["date"] == pd.Timestamp(current_date)]

        if signals.empty:
            continue

        signals = signals.merge(
            eligible[["symbol", "future_return_pct"]],
            on="symbol",
            how="inner",
            validate="one_to_one",
        )

        if not signals.empty:
            historical_signals.append(signals)

    if not historical_signals:
        print("No historical signals could be evaluated.")
        return

    results = pd.concat(historical_signals, ignore_index=True)

    # Select the highest-ranked signals for each date.
    top_results = (
        results.sort_values(
            ["date", "score", "return_7d_pct"],
            ascending=[True, False, False],
        )
        .groupby("date", group_keys=False)
        .head(TOP_COINS_PER_DAY)
        .copy()
    )

    if top_results.empty:
        print("No top-ranked signals were available to evaluate.")
        return

    model_mean = float(top_results["future_return_pct"].mean())
    model_median = float(top_results["future_return_pct"].median())

    # Randomly select from ALL eligible coins on the same evaluated
    # dates, matching the number of coins selected by the model.
    rng = np.random.default_rng(RANDOM_SEED)
    random_means = []

    for _ in range(RANDOM_TRIALS):
        trial_returns = []

        for date, model_day in top_results.groupby("date"):
            pool = eligible_pools.get(pd.Timestamp(date))

            if pool is None or pool.empty:
                continue

            pick_count = min(len(model_day), len(pool))
            picks = pool.sample(
                n=pick_count,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )

            trial_returns.extend(picks["future_return_pct"].tolist())

        if trial_returns:
            random_means.append(float(np.mean(trial_returns)))

    if not random_means:
        print("The random baseline could not be calculated.")
        return

    random_mean = float(np.mean(random_means))
    random_95 = float(np.percentile(random_means, 95))

    summary = pd.DataFrame([{
        "hold_days": HOLD_DAYS,
        "entry_rule": "Next daily open",
        "exit_rule": f"Close {HOLD_DAYS} days after entry",
        "top_coins_per_day": TOP_COINS_PER_DAY,
        "model_avg_forward_return_pct": model_mean,
        "model_median_forward_return_pct": model_median,
        "random_avg_forward_return_pct": random_mean,
        "random_95th_percentile_pct": random_95,
        "difference_vs_random_pct_points": model_mean - random_mean,
        "historical_signal_rows": len(results),
        "selected_signal_rows": len(top_results),
        "dates_evaluated": top_results["date"].nunique(),
        "random_trials": len(random_means),
    }])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nNimiQuant — Initial Historical Research")
    print("=" * 55)
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")
    print("\nImportant:")
    print("- Returns are gross; trading fees and slippage are excluded.")
    print("- The historical coin universe may contain survivorship bias.")
    print("- This test does not prove future profitability.")


if __name__ == "__main__":
    main()

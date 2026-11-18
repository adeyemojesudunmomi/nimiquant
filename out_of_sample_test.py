
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# NimiQuant — Out-of-Sample Strategy Test
# Built by Nimi | Project 1.0
# ============================================================

DATA_FILE = Path("data/prices.csv")
OUTPUT_FILE = Path("data/out_of_sample_results.csv")

HOLD_DAYS = 5
COINS_PER_PORTFOLIO = 5
RANDOM_TRIALS = 100
TRAIN_RATIO = 0.70

# Estimated entry and exit trading costs.
COST_PER_SIDE = 0.0015
RANDOM_SEED = 42


def net_return_pct(entry, exit_price):
    """Return percentage after estimated entry/exit costs."""
    return (
        (
            exit_price * (1 - COST_PER_SIDE)
            / (entry * (1 + COST_PER_SIDE))
        ) - 1
    ) * 100


def calculate_metrics(returns):
    """Calculate metrics from a sequence of portfolio returns."""
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "evaluations": 0,
            "mean_return_pct": np.nan,
            "median_return_pct": np.nan,
            "positive_evaluations_pct": np.nan,
            "worst_evaluation_pct": np.nan,
            "max_drawdown_pct": np.nan,
        }

    # Diagnostic equity curve: one equal-sized portfolio allocation
    # per evaluation. Overlapping holding periods mean this is NOT
    # a tradable daily equity curve.
    equity = np.cumprod(1 + values / 100)
    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], equity))
    )[1:]
    drawdowns = (equity / running_peak - 1) * 100

    return {
        "evaluations": len(values),
        "mean_return_pct": np.mean(values),
        "median_return_pct": np.median(values),
        "positive_evaluations_pct": np.mean(values > 0) * 100,
        "worst_evaluation_pct": np.min(values),
        "max_drawdown_pct": np.min(drawdowns),
    }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. Run data.py first."
        )

    prices = pd.read_csv(DATA_FILE)

    required = {
        "date", "symbol", "open", "high", "close", "quote_volume"
    }
    missing = required - set(prices.columns)

    if missing:
        raise ValueError(
            f"Missing columns in prices.csv: {sorted(missing)}"
        )

    prices["date"] = pd.to_datetime(prices["date"])

    prices = (
        prices.sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )

    grouped = prices.groupby("symbol", sort=False)

    # Features use only data available at the signal day's close.
    prices["history_count"] = grouped.cumcount() + 1
    prices["return_7d"] = grouped["close"].pct_change(7)
    prices["return_30d"] = grouped["close"].pct_change(30)

    prices["volume_avg_7d"] = grouped["quote_volume"].transform(
        lambda s: s.rolling(7).mean()
    )
    prices["volume_avg_30d"] = grouped["quote_volume"].transform(
        lambda s: s.rolling(30).mean()
    )
    prices["ma20"] = grouped["close"].transform(
        lambda s: s.rolling(20).mean()
    )

    volume_ratio = (
        prices["volume_avg_7d"]
        / prices["volume_avg_30d"].replace(0, np.nan)
    )

    prices["momentum_score"] = (
        (prices["return_7d"] > 0).astype(int)
        + (prices["return_30d"] > 0).astype(int)
        + (volume_ratio >= 1.5).astype(int)
        + (prices["close"] > prices["ma20"]).astype(int)
    )

    # Entry at next day's open; exit at close after HOLD_DAYS.
    prices["entry_open"] = grouped["open"].shift(-1)
    prices["exit_close"] = grouped["close"].shift(-(HOLD_DAYS + 1))

    valid = prices[
        (prices["history_count"] >= 60)
        & prices["entry_open"].notna()
        & prices["exit_close"].notna()
        & (prices["entry_open"] > 0)
        & (prices["exit_close"] > 0)
    ].copy()

    valid["forward_return_pct"] = net_return_pct(
        valid["entry_open"],
        valid["exit_close"]
    )

    # Split by DATE, not individual rows, to prevent the same date
    # from appearing in both training and test periods.
    all_dates = sorted(valid["date"].unique())

    if len(all_dates) < 20:
        raise ValueError(
            "Not enough historical dates for a reliable time split."
        )

    split_index = int(len(all_dates) * TRAIN_RATIO)
    split_index = max(1, min(split_index, len(all_dates) - 1))
    test_start_date = all_dates[split_index]

    test = valid[valid["date"] >= test_start_date].copy()

    rng = np.random.default_rng(RANDOM_SEED)

    momentum_returns = []
    random_trial_returns = [[] for _ in range(RANDOM_TRIALS)]
    test_dates = []

    for date, day in test.groupby("date", sort=True):
        eligible = day.dropna(
            subset=["momentum_score", "return_7d", "forward_return_pct"]
        )

        if len(eligible) < COINS_PER_PORTFOLIO:
            continue

        # Momentum selects the five highest scores.
        selected = eligible.sort_values(
            ["momentum_score", "return_7d"],
            ascending=False
        ).head(COINS_PER_PORTFOLIO)

        momentum_returns.append(
            selected["forward_return_pct"].mean()
        )

        # Random picks from the exact same eligible pool.
        for trial in range(RANDOM_TRIALS):
            random_picks = eligible.sample(
                n=COINS_PER_PORTFOLIO,
                replace=False,
                random_state=int(rng.integers(0, 2**32 - 1))
            )

            random_trial_returns[trial].append(
                random_picks["forward_return_pct"].mean()
            )

        test_dates.append(date)

    if not momentum_returns:
        raise ValueError(
            "No test dates had enough eligible coins."
        )

    # Each random trial covers the same test dates.
    random_means = [
        np.mean(trial)
        for trial in random_trial_returns
        if len(trial) == len(momentum_returns)
    ]

    if not random_means:
        raise ValueError("Random trials did not complete successfully.")

    all_random_returns = [
        value
        for trial in random_trial_returns
        for value in trial
    ]

    results = []

    momentum_metrics = calculate_metrics(momentum_returns)
    momentum_metrics["strategy"] = "Momentum"
    momentum_metrics["mean_of_trial_means_pct"] = np.nan
    results.append(momentum_metrics)

    random_metrics = calculate_metrics(all_random_returns)
    random_metrics["strategy"] = "Random"
    random_metrics["mean_of_trial_means_pct"] = np.mean(random_means)
    results.append(random_metrics)

    output = pd.DataFrame(results)

    # Put strategy first for readability.
    columns = ["strategy"] + [
        col for col in output.columns if col != "strategy"
    ]
    output = output[columns]

    numeric_columns = output.select_dtypes(
        include=["number"]
    ).columns
    output[numeric_columns] = output[numeric_columns].round(3)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\nNimiQuant — Out-of-Sample Test")
    print("=" * 72)
    print(f"Total historical dates: {len(all_dates)}")
    print(f"Training-period dates: {split_index}")
    print(f"Unseen test period starts: {pd.Timestamp(test_start_date).date()}")
    print(f"Test dates evaluated: {len(test_dates)}")
    print(f"Holding period: {HOLD_DAYS} days")
    print(f"Coins per portfolio: {COINS_PER_PORTFOLIO}")
    print(f"Random trials: {RANDOM_TRIALS}")
    print(f"Estimated cost per side: {COST_PER_SIDE * 100:.2f}%")
    print()
    print(output.to_string(index=False))

    if np.mean(momentum_returns) > np.mean(random_means):
        print("\nInitial result: Momentum beat the average random trial.")
    else:
        print("\nInitial result: Momentum did not beat the average random trial.")

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("\nImportant limitations:")
    print("- Historical coin-universe survivorship bias may remain.")
    print("- Five-day trades overlap across evaluation dates.")
    print("- The drawdown is diagnostic, not a live portfolio drawdown.")
    print("- Costs are estimates, not measured exchange fills.")
    print("- This test does not prove future profitability.")


if __name__ == "__main__":
    main()

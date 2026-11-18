
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# NimiQuant — Fair Strategy Comparison
# Built by Nimi | Project 1.0
# ============================================================

DATA_FILE = Path("data/prices.csv")
OUTPUT_FILE = Path("data/strategy_comparison_results.csv")

HOLD_DAYS = 5
COINS_PER_STRATEGY = 5
RANDOM_TRIALS = 100

# Estimated trading cost: 0.15% on entry and 0.15% on exit.
COST_PER_SIDE = 0.0015

RANDOM_SEED = 42


def calculate_net_return(entry, exit_price):
    """Calculate return after estimated entry and exit costs."""
    return (
        (
            exit_price * (1 - COST_PER_SIDE)
            / (entry * (1 + COST_PER_SIDE))
        ) - 1
    ) * 100


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. Run data.py first."
        )

    prices = pd.read_csv(DATA_FILE)

    required_columns = [
        "date", "symbol", "open", "high", "close", "quote_volume"
    ]

    missing = set(required_columns) - set(prices.columns)
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

    # --------------------------------------------------------
    # 1. Calculate historical features and future trade prices
    # --------------------------------------------------------

    grouped = prices.groupby("symbol", sort=False)

    prices["history_count"] = grouped.cumcount() + 1

    # Signal is observed at today's close.
    # Entry is the next day's open.
    # Exit is the close HOLD_DAYS days after entry.
    prices["entry_open"] = grouped["open"].shift(-1)
    prices["exit_close"] = grouped["close"].shift(-(HOLD_DAYS + 1))

    # Momentum indicators
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

    volume_ratio_momentum = (
        prices["volume_avg_7d"]
        / prices["volume_avg_30d"].replace(0, np.nan)
    )

    prices["momentum_score"] = (
        (prices["return_7d"] > 0).astype(int)
        + (prices["return_30d"] > 0).astype(int)
        + (volume_ratio_momentum >= 1.5).astype(int)
        + (prices["close"] > prices["ma20"]).astype(int)
    )

    # --------------------------------------------------------
    # 2. Calculate breakout indicators
    # --------------------------------------------------------

    # Today's candle must exceed the previous 20 candles' high.
    prices["previous_20d_high"] = grouped["high"].transform(
        lambda s: s.shift(1).rolling(20).max()
    )

    prices["previous_20d_volume"] = grouped["quote_volume"].transform(
        lambda s: s.shift(1).rolling(20).mean()
    )

    prices["volume_ratio"] = (
        prices["quote_volume"]
        / prices["previous_20d_volume"].replace(0, np.nan)
    )

    prices["breakout_signal"] = (
        (prices["close"] > prices["previous_20d_high"])
        & (prices["volume_ratio"] >= 1.5)
    )

    # --------------------------------------------------------
    # 3. Create one common pool for all strategies
    # --------------------------------------------------------

    valid = prices[
        (prices["history_count"] >= 60)
        & prices["entry_open"].notna()
        & prices["exit_close"].notna()
        & (prices["entry_open"] > 0)
        & (prices["exit_close"] > 0)
    ].copy()

    valid["net_return_pct"] = calculate_net_return(
        valid["entry_open"],
        valid["exit_close"]
    )

    date_groups = {}

    for date, day in valid.groupby("date"):
        if day["symbol"].nunique() >= COINS_PER_STRATEGY:
            date_groups[date] = day.copy()

    shared_dates = sorted(date_groups.keys())

    if not shared_dates:
        raise ValueError(
            "No shared evaluation dates. Check data/prices.csv."
        )

    # --------------------------------------------------------
    # 4. Evaluate Momentum, Breakout, and Random strategies
    # --------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    momentum_returns = []
    breakout_returns = []

    # Store a list of returns for each complete random trial.
    random_trial_returns = [
        [] for _ in range(RANDOM_TRIALS)
    ]

    for date in shared_dates:
        day = date_groups[date]

        # MOMENTUM: select up to five highest-scoring coins.
        momentum = (
            day.dropna(subset=["momentum_score", "return_7d"])
            .sort_values(
                ["momentum_score", "return_7d"],
                ascending=False
            )
            .head(COINS_PER_STRATEGY)
        )

        # Unfilled slots remain in cash, with zero return.
        momentum_return = (
            momentum["net_return_pct"].sum()
            / COINS_PER_STRATEGY
        )

        momentum_returns.append(momentum_return)

        # BREAKOUT: select up to five qualifying breakout coins.
        breakout = (
            day[day["breakout_signal"]]
            .dropna(subset=["volume_ratio"])
            .sort_values("volume_ratio", ascending=False)
            .head(COINS_PER_STRATEGY)
        )

        # Unfilled slots remain in cash, with zero return.
        breakout_return = (
            breakout["net_return_pct"].sum()
            / COINS_PER_STRATEGY
        )

        breakout_returns.append(breakout_return)

        # RANDOM: randomly choose five eligible coins per trial.
        eligible = day.dropna(subset=["net_return_pct"])

        for trial in range(RANDOM_TRIALS):
            picks = eligible.sample(
                n=COINS_PER_STRATEGY,
                replace=False,
                random_state=int(
                    rng.integers(0, 2**32 - 1)
                )
            )

            random_return = picks["net_return_pct"].mean()
            random_trial_returns[trial].append(random_return)

    # --------------------------------------------------------
    # 5. Summarize the results
    # --------------------------------------------------------

    def summarize(name, returns, dates_count):
        values = np.asarray(returns, dtype=float)

        return {
            "strategy": name,
            "dates_evaluated": dates_count,
            "average_daily_portfolio_return_pct": np.mean(values),
            "median_daily_portfolio_return_pct": np.median(values),
            "positive_days_pct": np.mean(values > 0) * 100,
            "period_average_return_95th_percentile_pct": np.nan
        }

    summary_rows = []

    summary_rows.append(
        summarize(
            "Momentum",
            momentum_returns,
            len(shared_dates)
        )
    )

    summary_rows.append(
        summarize(
            "Breakout",
            breakout_returns,
            len(shared_dates)
        )
    )

    # Each trial covers the same shared evaluation dates.
    random_period_means = [
        np.mean(trial_returns)
        for trial_returns in random_trial_returns
    ]

    # Pool returns to summarize typical random daily outcomes.
    all_random_returns = [
        value
        for trial_returns in random_trial_returns
        for value in trial_returns
    ]

    # IMPORTANT FIX:
    # Count unique dates, NOT dates multiplied by 100 trials.
    random_summary = summarize(
        "Random",
        all_random_returns,
        len(shared_dates)
    )

    random_summary[
        "period_average_return_95th_percentile_pct"
    ] = np.percentile(random_period_means, 95)

    summary_rows.append(random_summary)

    results = pd.DataFrame(summary_rows)

    numeric_columns = results.select_dtypes(
        include=["number"]
    ).columns

    results[numeric_columns] = results[numeric_columns].round(3)

    # --------------------------------------------------------
    # 6. Save and display
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nNimiQuant — Fair Strategy Comparison")
    print("=" * 75)
    print(f"Holding period: {HOLD_DAYS} days")
    print(f"Coins per strategy: {COINS_PER_STRATEGY}")
    print(f"Shared evaluation dates: {len(shared_dates)}")
    print(f"Random trials: {RANDOM_TRIALS}")
    print(f"Estimated costs per side: {COST_PER_SIDE * 100:.2f}%")
    print()
    print(results.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")

    print("\nImportant limitations:")
    print("- Historical survivorship bias may remain.")
    print("- Five-day trades overlap across evaluation dates.")
    print("- Trading costs are estimates, not measured exchange fills.")
    print("- This is not a full capital-based portfolio simulation.")
    print("- Historical results do not guarantee future profitability.")


if __name__ == "__main__":
    main()

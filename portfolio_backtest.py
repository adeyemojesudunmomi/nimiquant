
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# NimiQuant — Crypto Portfolio Backtest
# Built by Nimi | Project 1.0
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "prices.csv"
OUTPUT_DIR = BASE_DIR / "data"

STARTING_CAPITAL = 1000.00
MAX_POSITIONS = 5
HOLD_DAYS = 5
COST_PER_SIDE = 0.0015
MIN_SCORE = 2
RANDOM_TRIALS = 100
TRAINING_FRACTION = 0.70
RANDOM_SEED = 42


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {DATA_FILE}. Run py data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "date", "symbol", "open", "close", "quote_volume"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(
        df["date"], utc=True
    ).dt.tz_convert(None)

    for col in ["open", "close", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["date", "symbol", "open", "close", "quote_volume"]
    )

    df = df[
        (df["open"] > 0)
        & (df["close"] > 0)
        & (df["quote_volume"] >= 0)
    ]

    df = df.sort_values(["symbol", "date"])
    df = df.drop_duplicates(["symbol", "date"], keep="last")

    if df.empty:
        raise ValueError("No valid historical price data found.")

    return df.reset_index(drop=True)


def calculate_signals(df):
    pieces = []

    for symbol, group in df.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()

        group["return_7d"] = group["close"].pct_change(7)
        group["return_30d"] = group["close"].pct_change(30)

        volume_7d = group["quote_volume"].rolling(7).mean()
        volume_30d = group["quote_volume"].rolling(30).mean()

        group["volume_ratio"] = (
            volume_7d / volume_30d.replace(0, np.nan)
        )

        group["ma20"] = group["close"].rolling(20).mean()

        group["score"] = (
            (group["return_7d"] > 0).astype(int)
            + (group["return_30d"] > 0).astype(int)
            + (group["volume_ratio"] >= 1.5).astype(int)
            + (group["close"] > group["ma20"]).astype(int)
        )

        pieces.append(group)

    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def max_drawdown(equity_values):
    values = np.asarray(equity_values, dtype=float)

    if len(values) == 0:
        return 0.0

    peaks = np.maximum.accumulate(values)
    return float(((values / peaks) - 1).min() * 100)


def simulate_portfolio(data, dates, date_index=None,
                       mode="momentum", seed=0):
    if mode not in {"momentum", "random"}:
        raise ValueError("mode must be 'momentum' or 'random'")

    cash = float(STARTING_CAPITAL)
    positions = []
    equity_records = []
    trade_returns = []
    rng = np.random.default_rng(seed)

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])

    rows_by_date = {
        pd.Timestamp(date): group.set_index("symbol", drop=False)
        for date, group in data.groupby("date", sort=False)
    }

    dates = [pd.Timestamp(date) for date in dates]

    for i, date in enumerate(dates):
        today = rows_by_date.get(date, pd.DataFrame())

        previous_date = dates[i - 1] if i > 0 else None
        previous = (
            rows_by_date.get(previous_date, pd.DataFrame())
            if previous_date is not None
            else pd.DataFrame()
        )

        occupied = {p["symbol"] for p in positions}
        free_slots = MAX_POSITIONS - len(positions)

        if free_slots > 0 and not previous.empty and not today.empty:
            candidates = previous.copy()

            # Only coins with a usable entry price today are eligible.
            candidates = candidates[
                candidates.index.isin(today.index)
            ]
            candidates = candidates[
                np.isfinite(candidates["open"])
                & (candidates["open"] > 0)
            ]
            candidates = candidates[
                ~candidates.index.isin(occupied)
            ]

            # Fair comparison: both strategies use this same pool.
            candidates = candidates[
                (candidates["score"] >= MIN_SCORE)
                & candidates["return_7d"].notna()
                & candidates["volume_ratio"].notna()
            ]

            if mode == "momentum":
                candidates = candidates.sort_values(
                    ["score", "return_7d", "volume_ratio"],
                    ascending=False,
                )
                selected = list(candidates.index[:free_slots])

            else:
                pool = list(candidates.index)
                count = min(free_slots, len(pool))
                selected = (
                    list(rng.choice(pool, size=count, replace=False))
                    if count > 0 else []
                )

            for symbol in selected:
                if cash <= 0:
                    break

                entry_price = float(today.loc[symbol, "open"])
                slots_left = MAX_POSITIONS - len(positions)

                if (
                    not np.isfinite(entry_price)
                    or entry_price <= 0
                    or slots_left <= 0
                ):
                    continue

                budget = cash / slots_left
                quantity = budget / (
                    entry_price * (1 + COST_PER_SIDE)
                )

                cash -= budget

                positions.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "entry_date": date,
                    "exit_index": i + HOLD_DAYS,
                })

                occupied.add(symbol)

        remaining = []

        for position in positions:
            symbol = position["symbol"]

            if symbol not in today.index:
                raise ValueError(
                    f"Missing price for held coin {symbol} "
                    f"on {date.date()}. Cannot value portfolio safely."
                )

            close_price = float(today.loc[symbol, "close"])

            if not np.isfinite(close_price) or close_price <= 0:
                raise ValueError(
                    f"Invalid close price for {symbol} on {date.date()}."
                )

            should_exit = (
                i >= position["exit_index"]
                or i == len(dates) - 1
            )

            if should_exit:
                cash += (
                    position["quantity"]
                    * close_price
                    * (1 - COST_PER_SIDE)
                )

                net_return = (
                    close_price * (1 - COST_PER_SIDE)
                    / (
                        position["entry_price"]
                        * (1 + COST_PER_SIDE)
                    ) - 1
                ) * 100

                trade_returns.append(net_return)
            else:
                remaining.append(position)

        positions = remaining

        equity = cash
        for position in positions:
            symbol = position["symbol"]
            close_price = float(today.loc[symbol, "close"])
            equity += (
                position["quantity"]
                * close_price
                * (1 - COST_PER_SIDE)
            )

        equity_records.append({
            "date": date,
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
        })

    curve = pd.DataFrame(equity_records)

    if curve.empty:
        return {
            "final_equity": STARTING_CAPITAL,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profitable_days_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "curve": curve,
        }

    daily_returns = curve["equity"].pct_change().dropna()

    return {
        "final_equity": float(curve["equity"].iloc[-1]),
        "total_return_pct": (
            curve["equity"].iloc[-1] / STARTING_CAPITAL - 1
        ) * 100,
        "max_drawdown_pct": max_drawdown(curve["equity"]),
        "profitable_days_pct": (
            (daily_returns > 0).mean() * 100
            if len(daily_returns) else 0.0
        ),
        "trades": len(trade_returns),
        "win_rate_pct": (
            float((np.asarray(trade_returns) > 0).mean() * 100)
            if trade_returns else 0.0
        ),
        "curve": curve,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nNimiQuant — Crypto Portfolio Backtest")
    print("=" * 45)

    df = calculate_signals(load_data())
    dates = sorted(df["date"].unique())

    if len(dates) < 100:
        raise ValueError("Not enough historical dates to test.")

    split = int(len(dates) * TRAINING_FRACTION)
    test_dates = dates[split:]

    print(f"Cryptocurrencies loaded: {df['symbol'].nunique()}")
    print(f"Historical dates: {len(dates)}")
    print(f"Out-of-sample test dates: {len(test_dates)}")
    print(f"Test begins: {pd.Timestamp(test_dates[0]).date()}")
    print(f"Starting capital: ${STARTING_CAPITAL:,.2f}")
    print(f"Maximum positions: {MAX_POSITIONS}")
    print(f"Holding period: {HOLD_DAYS} days")
    print(f"Fee per side: {COST_PER_SIDE * 100:.2f}%")
    print(f"Random trials: {RANDOM_TRIALS}")

    momentum = simulate_portfolio(
        df, test_dates, mode="momentum"
    )

    random_results = []

    for trial in range(RANDOM_TRIALS):
        result = simulate_portfolio(
            df,
            test_dates,
            mode="random",
            seed=RANDOM_SEED + trial,
        )

        random_results.append({
            "trial": trial + 1,
            "total_return_pct": result["total_return_pct"],
            "max_drawdown_pct": result["max_drawdown_pct"],
            "final_equity": result["final_equity"],
        })

    random_df = pd.DataFrame(random_results)

    summary = pd.DataFrame([
        {
            "strategy": "Momentum",
            "final_equity": momentum["final_equity"],
            "total_return_pct": momentum["total_return_pct"],
            "max_drawdown_pct": momentum["max_drawdown_pct"],
            "profitable_days_pct": momentum["profitable_days_pct"],
            "trades": momentum["trades"],
            "win_rate_pct": momentum["win_rate_pct"],
        },
        {
            "strategy": "Random average",
            "final_equity": random_df["final_equity"].mean(),
            "total_return_pct": random_df["total_return_pct"].mean(),
            "max_drawdown_pct": random_df["max_drawdown_pct"].mean(),
            "profitable_days_pct": np.nan,
            "trades": np.nan,
            "win_rate_pct": np.nan,
        },
        {
            "strategy": "Random median",
            "final_equity": random_df["final_equity"].median(),
            "total_return_pct": random_df["total_return_pct"].median(),
            "max_drawdown_pct": random_df["max_drawdown_pct"].median(),
            "profitable_days_pct": np.nan,
            "trades": np.nan,
            "win_rate_pct": np.nan,
        },
    ])

    print("\nRESULTS")
    print("=" * 45)
    print(summary.round(3).to_string(index=False))

    print("\nRandom trial return percentiles:")
    print(
        random_df["total_return_pct"]
        .quantile([0.05, 0.50, 0.95])
        .rename(index={
            0.05: "5th percentile",
            0.50: "Median",
            0.95: "95th percentile",
        })
        .round(3)
        .to_string()
    )

    momentum["curve"].to_csv(
        OUTPUT_DIR / "portfolio_equity.csv", index=False
    )
    summary.to_csv(
        OUTPUT_DIR / "portfolio_backtest_results.csv", index=False
    )
    random_df.to_csv(
        OUTPUT_DIR / "portfolio_random_trials.csv", index=False
    )

    print("\nSaved results to the data folder.")
    print("\nThis is a historical simulation, not a profit guarantee.")
    print("Survivorship bias and real-world slippage remain limitations.")


if __name__ == "__main__":
    main()


"""
NimiQuant — Crypto Intelligence Engine
Project 1.0 | Built by Nimi

Purpose:
Identify higher-quality momentum and breakout setups,
with risk-based position sizing.
Signals are candidates, not profit guarantees.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==================== SETTINGS ====================

DATA_FILE = Path("data/prices.csv")
OUTPUT_FILE = Path("data/signals.csv")

MIN_DAILY_LIQUIDITY = 5_000_000
MIN_VOLUME_RATIO = 1.20
MIN_SCORE = 5

MAX_7D_RETURN = 0.30
MAX_RSI = 70

ATR_STOP_MULTIPLIER = 1.5
REWARD_RISK_RATIO = 2.0

ACCOUNT_BALANCE = 1000.00
RISK_PER_TRADE = 0.01
MAX_ALLOCATION = 0.20


# ==================== LOAD DATA ====================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run py data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "date", "symbol", "open", "high",
        "low", "close", "quote_volume"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in prices.csv: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "open", "high", "low", "close", "quote_volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["date", "symbol", *numeric_cols]
    )

    df = df[
        (df["close"] > 0)
        & (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["quote_volume"] >= 0)
    ].copy()

    df = df.sort_values(["symbol", "date"])
    df = df.drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )

    if df.empty:
        raise ValueError("No valid price records found.")

    return df


# ==================== FEATURES ====================

def calculate_features(df):
    frames = []

    for symbol, group in df.groupby("symbol", sort=False):
        g = group.copy().sort_values("date")

        close = g["close"]
        high = g["high"]
        low = g["low"]
        volume = g["quote_volume"]

        # Momentum
        g["return_7d"] = close.pct_change(7)
        g["return_30d"] = close.pct_change(30)

        # Trend
        g["sma20"] = close.rolling(20).mean()
        g["sma50"] = close.rolling(50).mean()

        # Volume activity
        g["avg_volume_7d"] = volume.rolling(7).mean()
        g["avg_volume_30d"] = (
            volume.shift(1).rolling(30).mean()
        )

        g["volume_ratio"] = (
            g["avg_volume_7d"]
            / g["avg_volume_30d"].replace(0, np.nan)
        )

        # Breakout above the previous 20 completed candles
        g["previous_high_20d"] = (
            high.shift(1).rolling(20).max()
        )

        g["breakout"] = close > g["previous_high_20d"]

        # RSI(14)
        change = close.diff()
        gains = change.clip(lower=0)
        losses = -change.clip(upper=0)

        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        g["rsi14"] = 100 - (100 / (1 + rs))

        # Sustained rises with no losses have RSI = 100
        g.loc[
            (avg_loss == 0) & (avg_gain > 0), "rsi14"
        ] = 100

        # ATR(14)
        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        g["atr14"] = true_range.rolling(14).mean()
        g["atr_pct"] = g["atr14"] / close

        frames.append(g)

    return pd.concat(frames, ignore_index=True)


# ==================== SCORING & RISK ====================

def score_latest(latest):
    x = latest.copy()

    # Seven quality checks
    x["score"] = (
        (x["close"] > x["sma20"]).astype(int)
        + (x["sma20"] > x["sma50"]).astype(int)
        + (x["return_7d"] > 0).astype(int)
        + (x["return_30d"] > 0).astype(int)
        + (x["volume_ratio"] >= MIN_VOLUME_RATIO).astype(int)
        + x["breakout"].astype(int)
        + (
            (x["rsi14"] >= 50)
            & (x["rsi14"] < MAX_RSI)
        ).astype(int)
    )

    # Strict eligibility filters
    eligible = x[
        (x["close"] > x["sma20"])
        & (x["sma20"] > x["sma50"])
        & (x["return_7d"] > 0)
        & (x["return_30d"] > 0)
        & (x["return_7d"] <= MAX_7D_RETURN)
        & (x["avg_volume_30d"] >= MIN_DAILY_LIQUIDITY)
        & (x["volume_ratio"] >= MIN_VOLUME_RATIO)
        & (x["rsi14"] >= 50)
        & (x["rsi14"] < MAX_RSI)
        & (x["atr14"] > 0)
        & (x["score"] >= MIN_SCORE)
    ].copy()

    if eligible.empty:
        return eligible

    eligible["setup_type"] = np.where(
        eligible["breakout"],
        "Confirmed breakout",
        "Trend continuation",
    )

    # Reference entry based on latest completed close
    eligible["reference_entry"] = eligible["close"]

    # ATR-based stop and 2:1 reward/risk target
    eligible["stop_loss"] = (
        eligible["reference_entry"]
        - ATR_STOP_MULTIPLIER * eligible["atr14"]
    )

    eligible["risk_per_coin"] = (
        eligible["reference_entry"] - eligible["stop_loss"]
    )

    eligible["target_1"] = (
        eligible["reference_entry"]
        + REWARD_RISK_RATIO * eligible["risk_per_coin"]
    )

    eligible["stop_distance_pct"] = (
        eligible["risk_per_coin"]
        / eligible["reference_entry"]
    ) * 100

    eligible["target_gain_pct"] = (
        (
            eligible["target_1"]
            - eligible["reference_entry"]
        )
        / eligible["reference_entry"]
    ) * 100

    # Risk-based position sizing
    eligible["risk_budget_usd"] = (
        ACCOUNT_BALANCE * RISK_PER_TRADE
    )

    # Size required to risk 1% if the stop executes as planned
    eligible["position_size_coins"] = (
        eligible["risk_budget_usd"]
        / eligible["risk_per_coin"]
    )

    eligible["position_value_usd"] = (
        eligible["position_size_coins"]
        * eligible["reference_entry"]
    )

    # Never allocate more than 20% of the configured account
    eligible["max_position_value_usd"] = (
        ACCOUNT_BALANCE * MAX_ALLOCATION
    )

    eligible["position_value_usd"] = eligible[
        ["position_value_usd", "max_position_value_usd"]
    ].min(axis=1)

    # Recalculate quantity and planned risk after allocation cap
    eligible["position_size_coins"] = (
        eligible["position_value_usd"]
        / eligible["reference_entry"]
    )

    eligible["planned_risk_usd"] = (
        eligible["position_size_coins"]
        * eligible["risk_per_coin"]
    )

    eligible = eligible.sort_values(
        ["score", "breakout", "return_30d", "volume_ratio"],
        ascending=[False, False, False, False],
    )

    return eligible


# ==================== RUN SCANNER ====================

def main():
    print("\nNimiQuant — Crypto Intelligence Engine")
    print("=" * 48)

    df = load_data()

    print(f"Coins loaded: {df['symbol'].nunique()}")
    print(f"Price records: {len(df):,}")

    features = calculate_features(df)

    latest_date = features["date"].max()

    latest = features[
        features["date"] == latest_date
    ].copy()

    latest = latest.dropna(
        subset=[
            "return_7d",
            "return_30d",
            "sma20",
            "sma50",
            "volume_ratio",
            "avg_volume_30d",
            "previous_high_20d",
            "rsi14",
            "atr14",
        ]
    )

    signals = score_latest(latest)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUTPUT_FILE, index=False)

    print(f"Latest data date: {latest_date.date()}")
    print(f"Coins evaluated: {len(latest)}")
    print(f"Qualifying signals: {len(signals)}")
    print(f"Account balance assumption: ${ACCOUNT_BALANCE:,.2f}")
    print(f"Risk budget per trade: ${ACCOUNT_BALANCE * RISK_PER_TRADE:.2f}")
    print(f"Maximum allocation per coin: {MAX_ALLOCATION:.0%}")
    print(f"Saved to: {OUTPUT_FILE}")

    if signals.empty:
        print("\nNo high-quality setups right now.")
        print("The scanner will not force a trade.")
        return

    display_cols = [
        "symbol",
        "score",
        "setup_type",
        "reference_entry",
        "stop_loss",
        "target_1",
        "stop_distance_pct",
        "position_value_usd",
        "planned_risk_usd",
    ]

    display = signals[display_cols].head(15).copy()

    print("\nTOP NIMIQUANT SIGNALS")
    print("-" * 110)
    print(display.round(4).to_string(index=False))

    print("\nPosition values and risk are estimates in USD.")
    print("Stops may execute at worse prices during fast markets.")
    print("Fees and slippage can increase actual losses.")
    print("Signals are not guaranteed to be profitable.")


if __name__ == "__main__":
    main()


"""
NimiQuant — Breakout Research Model
Project 1.0 | Built by Nimi

Identifies potential upward breakouts for historical testing.
This is an experimental research model, not a buy recommendation.
"""

from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/prices.csv")
OUTPUT_FILE = Path("data/breakout_signals.csv")

BREAKOUT_DAYS = 20
VOLUME_DAYS = 20
MIN_VOLUME_RATIO = 1.5


def calculate_breakouts(df):
    required = {
        "date", "symbol", "open", "high", "low", "close", "quote_volume"
    }

    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    results = []

    for symbol, coin in df.groupby("symbol", sort=False):
        coin = coin.copy()

        # Exclude today's candle from the previous-high calculation.
        coin["previous_high"] = (
            coin["high"].shift(1).rolling(BREAKOUT_DAYS).max()
        )

        # Compare today's volume with the previous 20 days.
        coin["previous_avg_volume"] = (
            coin["quote_volume"].shift(1).rolling(VOLUME_DAYS).mean()
        )

        coin["volume_ratio"] = (
            coin["quote_volume"] / coin["previous_avg_volume"]
        )

        # Require a close above resistance, not just an intraday wick.
        coin["breakout"] = (
            (coin["close"] > coin["previous_high"])
            & (coin["volume_ratio"] >= MIN_VOLUME_RATIO)
            & (coin["previous_avg_volume"] > 0)
        )

        signals = coin.loc[coin["breakout"]].copy()

        for _, row in signals.iterrows():
            results.append({
                "date": row["date"],
                "symbol": symbol,
                "price": row["close"],
                "previous_high": row["previous_high"],
                "volume_ratio": round(row["volume_ratio"], 2),
            })

    columns = [
        "date", "symbol", "price", "previous_high", "volume_ratio"
    ]

    return pd.DataFrame(results, columns=columns)


def main():
    if not DATA_FILE.exists():
        print("Price data not found. Run data.py first.")
        return

    prices = pd.read_csv(DATA_FILE, parse_dates=["date"])
    signals = calculate_breakouts(prices)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUTPUT_FILE, index=False)

    print("\nNimiQuant — Experimental Breakout Signals")
    print("=" * 55)

    if signals.empty:
        print("No breakout signals found.")
    else:
        print(f"Breakout signals found: {len(signals):,}")
        print(
            signals.sort_values("date", ascending=False)
            .head(15)
            .to_string(index=False)
        )

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("Signals are candidates for research, not guaranteed winners.")


if __name__ == "__main__":
    main()

"""
NimiQuant — Live Phantom Wallet Breakout Scanner
Project 2.0 | Built by Nimi
"""

from pathlib import Path
import pandas as pd

DATA_FILE = Path("data/solana_prices.csv")
BREAKOUT_DAYS = 20
VOLUME_DAYS = 20
MIN_VOLUME_RATIO = 1.5


def scan_live_breakouts():
    if not DATA_FILE.exists():
        print("Solana data not found. Run `py data.py` first.")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values(["ca", "date"])
    latest_market_date = df["date"].max()

    signals = []

    for ca, coin in df.groupby("ca", sort=False):
        coin = coin.copy()
        if len(coin) < BREAKOUT_DAYS + 1:
            continue

        coin["previous_high"] = coin["high"].shift(1).rolling(BREAKOUT_DAYS).max()
        coin["previous_avg_volume"] = coin["quote_volume"].shift(1).rolling(VOLUME_DAYS).mean()
        coin["volume_ratio"] = coin["quote_volume"] / coin["previous_avg_volume"]

        # Check ONLY the latest candle for an active breakout today
        today_row = coin.iloc[-1]
        if (
            today_row["date"] == latest_market_date
            and today_row["close"] > today_row["previous_high"]
            and today_row["volume_ratio"] >= MIN_VOLUME_RATIO
            and today_row["previous_avg_volume"] > 0
        ):
            signals.append({
                "symbol": today_row["symbol"],
                "ca": ca,
                "date": today_row["date"].strftime("%Y-%m-%d"),
                "price": today_row["close"],
                "volume_24h": round(today_row["quote_volume"], 0),
                "volume_ratio": round(today_row["volume_ratio"], 2),
            })

    print("\n" + "=" * 65)
    print(f"NimiQuant — Live Phantom Wallet Signals ({latest_market_date.date()})")
    print("=" * 65)

    if not signals:
        print("No fresh 20-day breakouts triggered today. Patience protects capital.")
        return

    signals_df = pd.DataFrame(signals).sort_values("volume_ratio", ascending=False)
    print(f"Active Breakouts Today: {len(signals_df)}\n")

    for _, row in signals_df.iterrows():
        print(f"Token        : {row['symbol']}")
        print(f"Price        : ${row['price']:.6f}")
        print(f"24h Volume   : ${row['volume_24h']:,.0f} ({row['volume_ratio']}x 20d avg)")
        print(f"Phantom CA   : {row['ca']}")
        print("-" * 65)


if __name__ == "__main__":
    scan_live_breakouts()
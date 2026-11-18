"""
NimiQuant — Solana DEX Market Data Engine
Project 2.0 | Built by Nimi

Fetches live trending & high-volume Solana tokens and daily OHLCV history
from GeckoTerminal (Raydium/Orca/Meteora) with exact Contract Addresses (CAs).
Includes automatic rate-limit handling (HTTP 429 retry) for free-tier stability.

Install:
    pip install requests pandas

Run:
    py data.py

Output:
    data/solana_prices.csv
"""

import time
from pathlib import Path
import pandas as pd
import requests

BASE_URL = "https://api.geckoterminal.com/api/v2"
DATA_FILE = Path("data/solana_prices.csv")
PAGES_TO_FETCH = 3  # 20 pools per page = top 60 active Solana pools
MIN_LIQUIDITY_USD = 100_000
MIN_DAILY_VOLUME = 250_000

HEADERS = {"Accept": "application/json;version=20230203"}
COLUMNS = ["date", "symbol", "ca", "open", "high", "low", "close", "quote_volume"]


def get_top_solana_pools():
    pools = []
    seen_cas = set()

    for page in range(1, PAGES_TO_FETCH + 1):
        try:
            resp = requests.get(
                f"{BASE_URL}/networks/solana/trending_pools",
                params={"page": page},
                headers=HEADERS,
                timeout=15,
            )
            if resp.status_code == 429:
                print(f"[Rate Limit] Waiting 10s before retrying trending page {page}...")
                time.sleep(10)
                resp = requests.get(
                    f"{BASE_URL}/networks/solana/trending_pools",
                    params={"page": page},
                    headers=HEADERS,
                    timeout=15,
                )

            resp.raise_for_status()
            data = resp.json().get("data", [])

            for item in data:
                attr = item.get("attributes", {})
                rel = item.get("relationships", {})

                base_id = rel.get("base_token", {}).get("data", {}).get("id", "")
                ca = base_id.replace("solana_", "")

                # Skip Wrapped SOL itself or duplicate CAs
                if not ca or ca == "So11111111111111111111111111111111111111112" or ca in seen_cas:
                    continue

                reserve = float(attr.get("reserve_in_usd") or 0)
                vol_24h = float(attr.get("volume_usd", {}).get("h24") or 0)
                name = attr.get("name", "UNKNOWN").split("/")[0].strip()
                pool_address = attr.get("address")

                if reserve >= MIN_LIQUIDITY_USD and vol_24h >= MIN_DAILY_VOLUME and pool_address:
                    seen_cas.add(ca)
                    pools.append({
                        "symbol": name,
                        "ca": ca,
                        "pool_address": pool_address,
                    })

            time.sleep(1.5)
        except Exception as e:
            print(f"[Warning] Page {page} fetch failed: {e}")

    return pools


def download_pool_ohlcv(pool_info, limit=60):
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{BASE_URL}/networks/solana/pools/{pool_info['pool_address']}/ohlcv/day",
                params={"limit": limit, "currency": "usd"},
                headers=HEADERS,
                timeout=15,
            )
            if resp.status_code == 429:
                print(f"   [Rate Limit] Waiting 12s before retrying {pool_info['symbol']}...")
                time.sleep(12)
                continue

            resp.raise_for_status()
            ohlcv_list = resp.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])

            if not ohlcv_list:
                return pd.DataFrame(columns=COLUMNS)

            df = pd.DataFrame(
                ohlcv_list,
                columns=["timestamp", "open", "high", "low", "close", "quote_volume"],
            )
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
            df["symbol"] = pool_info["symbol"]
            df["ca"] = pool_info["ca"]

            for col in ["open", "high", "low", "close", "quote_volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df[COLUMNS].dropna().sort_values("date")
        except Exception as e:
            if attempt == 2:
                print(f"   [Error] {pool_info['symbol']}: {e}")
            time.sleep(2)

    return pd.DataFrame(columns=COLUMNS)


def main():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    print("Scanning live Solana DEX pools via GeckoTerminal...")

    pools = get_top_solana_pools()
    if not pools:
        print("[Error] Could not retrieve Solana pools. Check internet connection.")
        return

    print(f"Found {len(pools)} liquid Solana tokens. Downloading daily candles...")
    frames = []

    for idx, pool in enumerate(pools, start=1):
        df = download_pool_ohlcv(pool, limit=60)
        if not df.empty:
            frames.append(df)
        print(f"[{idx}/{len(pools)}] {pool['symbol']} ({pool['ca'][:6]}...): {len(df)} days")
        time.sleep(2.2)  # Stay safely under GeckoTerminal's 30 req/min free limit

    if not frames:
        print("No candle data downloaded.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ca", "date"], keep="last").sort_values(["symbol", "date"])
    combined.to_csv(DATA_FILE, index=False)

    print("\nNimiQuant Solana DEX update complete!")
    print(f"Tokens tracked: {combined['ca'].nunique()}")
    print(f"Latest candle : {combined['date'].max().date()}")
    print(f"Saved to      : {DATA_FILE}")


if __name__ == "__main__":
    main()
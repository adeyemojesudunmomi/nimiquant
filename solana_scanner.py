
"""
NimiQuant — Solana Opportunity Scanner
Project 1.0 | Built by Nimi
"""

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


# ---------------- SETTINGS ----------------

API = "https://api.dexscreener.com"
OUTPUT = Path("data/solana_signals.csv")

SEARCH_QUERIES = [
    "SOL/USDC",
    "BONK",
    "WIF",
    "JUP",
    "RAY",
    "PUMP",
    "POPCAT",
    "FARTCOIN",
    "USDC",
    "SOL",
]

MIN_LIQUIDITY = 50_000
MIN_VOLUME_24H = 50_000
MIN_TRANSACTIONS = 100
MIN_AGE_HOURS = 6
MAX_AGE_HOURS = 30 * 24
MAX_GAIN_24H = 100
MIN_SCORE = 4

DELAY = 0.15

EXCLUDED_SYMBOLS = {
    "USDC", "USDT", "USDS", "DAI", "PYUSD",
    "SOL", "WSOL", "USD1", "FDUSD", "TUSD",
}


# ---------------- API HELPERS ----------------

def fetch_json(url):
    request = Request(
        url,
        headers={"User-Agent": "NimiQuant/1.0"},
    )

    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def number(value, default=0.0):
    try:
        result = float(value)
        return result if pd.notna(result) else default
    except (TypeError, ValueError):
        return default


def discover_pairs():
    """Discover pairs using search and recent token profiles."""

    pairs_by_address = {}

    # Search multiple terms to broaden discovery.
    for query in SEARCH_QUERIES:
        try:
            url = f"{API}/latest/dex/search?q={quote(query)}"
            data = fetch_json(url)

            for pair in data.get("pairs") or []:
                if pair.get("chainId") != "solana":
                    continue

                address = pair.get("pairAddress")
                if address:
                    pairs_by_address[address] = pair

            print(f"Search '{query}': "
                  f"{len(data.get('pairs') or [])} pairs")

        except (HTTPError, URLError, TimeoutError,
                ValueError, OSError) as exc:
            print(f"Search '{query}' failed: {exc}")

        time.sleep(DELAY)

    # Add pairs associated with recently profiled Solana tokens.
    try:
        profiles = fetch_json(
            f"{API}/token-profiles/latest/v1"
        )

        addresses = list(dict.fromkeys(
            item.get("tokenAddress")
            for item in profiles
            if item.get("chainId") == "solana"
            and item.get("tokenAddress")
        ))

        print(f"Recent Solana profile addresses: {len(addresses)}")

        # DexScreener supports up to 30 addresses per batch.
        for start in range(0, len(addresses), 30):
            batch = addresses[start:start + 30]
            joined = ",".join(quote(a, safe="") for a in batch)

            data = fetch_json(
                f"{API}/tokens/v1/solana/{joined}"
            )

            for pair in data if isinstance(data, list) else []:
                if pair.get("chainId") != "solana":
                    continue

                address = pair.get("pairAddress")
                if address:
                    pairs_by_address[address] = pair

            time.sleep(DELAY)

    except (HTTPError, URLError, TimeoutError,
            ValueError, OSError) as exc:
        print(f"Profile discovery failed: {exc}")

    return list(pairs_by_address.values())


# ---------------- ANALYSIS ----------------

def analyse_pair(pair, failures):
    base = pair.get("baseToken") or {}
    address = base.get("address", "")
    symbol = base.get("symbol", "UNKNOWN")

    if not address:
        failures["missing_address"] += 1
        return None

    if symbol.upper() in EXCLUDED_SYMBOLS:
        failures["excluded_base_asset"] += 1
        return None

    price = number(pair.get("priceUsd"), -1)
    liquidity = number((pair.get("liquidity") or {}).get("usd"))
    volume = number((pair.get("volume") or {}).get("h24"))

    changes = pair.get("priceChange") or {}
    change_6h = number(changes.get("h6"))
    change_24h = number(changes.get("h24"))

    txns = (pair.get("txns") or {}).get("h24") or {}
    buys = int(number(txns.get("buys")))
    sells = int(number(txns.get("sells")))
    transaction_count = buys + sells

    created_at = pair.get("pairCreatedAt")

    if not created_at:
        failures["missing_pair_age"] += 1
        return None

    age_hours = (
        datetime.now(timezone.utc)
        - datetime.fromtimestamp(
            created_at / 1000, tz=timezone.utc
        )
    ).total_seconds() / 3600

    # Count the first failed filter for each pair.
    if price <= 0:
        failures["invalid_price"] += 1
        return None

    if liquidity < MIN_LIQUIDITY:
        failures["low_liquidity"] += 1
        return None

    if volume < MIN_VOLUME_24H:
        failures["low_volume"] += 1
        return None

    if transaction_count < MIN_TRANSACTIONS:
        failures["low_transactions"] += 1
        return None

    if age_hours < MIN_AGE_HOURS:
        failures["too_new"] += 1
        return None

    if age_hours > MAX_AGE_HOURS:
        failures["too_old"] += 1
        return None

    if change_24h <= 0 or change_24h > MAX_GAIN_24H:
        failures["price_change_filter"] += 1
        return None

    score = (
        int(liquidity >= 250_000)
        + int(volume >= 250_000)
        + int(change_6h > 0)
        + int(change_24h > 0)
        + int(buys > sells)
        + int(volume / liquidity >= 0.5)
        + int(age_hours >= 24)
    )

    if score < MIN_SCORE:
        failures["low_score"] += 1
        return None

    failures["passed"] += 1

    return {
        "name": base.get("name", "Unknown"),
        "symbol": symbol,
        "network": "Solana",
        "token_address": address,
        "price_usd": price,
        "score": score,
        "change_6h_pct": change_6h,
        "change_24h_pct": change_24h,
        "liquidity_usd": liquidity,
        "volume_24h_usd": volume,
        "buys_24h": buys,
        "sells_24h": sells,
        "pair_age_hours": round(age_hours, 1),
        "dex_url": pair.get("url", ""),
        "solscan_url": f"https://solscan.io/token/{address}",
    }


# ---------------- MAIN ----------------

def main():
    print("\nNimiQuant — Solana Opportunity Scanner")
    print("=" * 48)

    print("Discovering Solana trading pairs...")
    pairs = discover_pairs()

    print(f"\nUnique Solana pairs discovered: {len(pairs)}")

    failures = Counter()
    candidates = []

    for pair in pairs:
        result = analyse_pair(pair, failures)
        if result:
            candidates.append(result)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(candidates)

    if not df.empty:
        # Keep the most liquid pair for each token.
        df = (
            df.sort_values(
                ["score", "liquidity_usd", "volume_24h_usd"],
                ascending=False,
            )
            .drop_duplicates(subset=["token_address"])
            .reset_index(drop=True)
        )

    df.to_csv(OUTPUT, index=False)

    print("\nFILTER DIAGNOSTICS")
    print("-" * 35)

    if failures:
        for reason, count in sorted(failures.items()):
            print(f"{reason}: {count}")
    else:
        print("No pairs reached the analysis stage.")

    print(f"\nQualifying unique tokens: {len(df)}")
    print(f"Saved to: {OUTPUT}")

    if df.empty:
        print("\nNo candidates passed all filters.")
        print("Use the diagnostics above to identify the bottleneck.")
        return

    columns = [
        "name", "symbol", "score", "price_usd",
        "change_24h_pct", "liquidity_usd",
        "volume_24h_usd", "token_address",
    ]

    print("\nTOP SOLANA CANDIDATES")
    print("-" * 100)
    print(df[columns].head(15).round(4).to_string(index=False))

    print("\nToken addresses and verification links are saved in the CSV.")
    print("A qualifying score does not prove a token is safe or profitable.")


if __name__ == "__main__":
    main()

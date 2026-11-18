"""
NimiQuant Crypto Intelligence Engine — Multi-Chain Active Trade Scanner + Live PnL, Telegram Commands & CSV Export
Project 6.0 | Built by Nimi

High-precision dual-timeframe verification (exact Daily + 4-Hour OHLCV candles),
hybrid Trending + New Pool discovery for low-cap gems, low-cap conviction scoring,
instant HTML-formatted Telegram trade alerts, live PnL tracking, two-way Telegram
command polling, and full CSV trade history export.

Run:
    py -m streamlit run app.py
"""

import html
import os
import time
import pandas as pd
import requests
import streamlit as st

# =========================
# API & TELEGRAM CONFIG
# =========================
BASE_URL = "https://api.geckoterminal.com/api/v2"
HEADERS = {"Accept": "application/json;version=20230203"}

# Loaded securely from your private .streamlit/secrets.toml file
DEFAULT_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))

NATIVE_WRAPPED_CAS = {
    "So11111111111111111111111111111111111111112",  # WSOL
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",   # WETH (ETH)
    "0x4200000000000000000000000000000000000006",   # WETH (Base / OP)
    "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c",   # WBNB (BSC)
    "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",   # WETH (Arbitrum)
    "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270",   # WPOL (Polygon)
}

NETWORK_WALLET_GUIDE = {
    "solana": ("Solana", "Phantom, Solflare, Backpack, Trust Wallet"),
    "base": ("Base", "Coinbase Wallet, MetaMask, Trust Wallet, Rabby"),
    "eth": ("Ethereum", "MetaMask, Trust Wallet, Rabby, Rainbow, Coinbase"),
    "bsc": ("BNB Chain (BSC)", "Trust Wallet, MetaMask, Binance Web3, Rabby"),
    "arbitrum": ("Arbitrum", "MetaMask, Rabby, Trust Wallet, Coinbase Wallet"),
    "polygon_pos": ("Polygon", "MetaMask, Trust Wallet, Phantom, Rabby"),
    "optimism": ("Optimism", "MetaMask, Rabby, Coinbase Wallet, Trust Wallet"),
    "avax": ("Avalanche", "Core, MetaMask, Trust Wallet, Rabby"),
    "sui-network": ("Sui", "Sui Wallet, Suiet, Phantom, Backpack"),
    "ton": ("TON", "Tonkeeper, MyTonWallet, Trust Wallet"),
}


# =========================
# TELEGRAM NOTIFICATION ENGINE
# =========================
def send_telegram_message(bot_token, chat_id, html_text):
    if not bot_token or not chat_id:
        return False, "Missing Bot Token or Chat ID in secrets.toml."

    url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
    payload = {
        "chat_id": str(chat_id).strip(),
        "text": html_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=12)
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            return True, "Delivered"
        return False, data.get("description", f"HTTP {resp.status_code}")
    except Exception as e:
        return False, str(e)


def format_trade_alert(signal):
    sym = html.escape(str(signal["Symbol"]))
    net = html.escape(str(signal["Network"]))
    tier = html.escape(str(signal["Tier"]))
    cap_tag = html.escape(str(signal["Cap Tier"]))
    wallets = html.escape(str(signal["Compatible Wallets"]))
    ca = html.escape(str(signal["Contract Address"]))
    mcap_str = f"${signal['Market Cap ($)']:,.0f}" if signal["Market Cap ($)"] > 0 else "Unverified / Early Gem"

    return (
        f"🚨 <b>NIMIQUANT INTELLIGENCE ALERT</b> 🚨\n\n"
        f"🔥 <b>{sym}</b>  [<code>{net}</code>]  •  {cap_tag}\n"
        f"📊 <b>Setup:</b> {tier}\n"
        f"⭐ <b>Conviction Score:</b> {signal['Score']} / 100\n\n"
        f"💵 <b>Entry Price:</b> <code>${signal['Price ($)']:.6f}</code> ({signal['24h Change (%)']}%)\n"
        f"💎 <b>Market Cap / FDV:</b> {mcap_str}\n"
        f"📈 <b>Volume Surge:</b> {signal['Volume Surge']}x avg\n"
        f"💧 <b>Liquidity:</b> ${signal['Liquidity ($)']:,.0f} | <b>24h Vol:</b> ${signal['24h Volume ($)']:,.0f}\n\n"
        f"🛡️ <b>Risk Management Plan:</b>\n"
        f"• <b>Stop-Loss:</b> <code>${signal['Stop-Loss']}</code>\n"
        f"• <b>Take-Profit:</b> <code>${signal['Take-Profit']}</code>\n\n"
        f"💼 <b>Supported Wallets:</b> {wallets}\n\n"
        f"📋 <b>Contract Address (Tap to Copy):</b>\n"
        f"<code>{ca}</code>"
    )


# =========================
# HIGH-PRECISION DATA FETCHING
# =========================
def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_hybrid_pools(
    session,
    selected_network="all",
    scan_mode="hybrid",
    pages=3,
    min_liq=25_000,
    min_vol=50_000,
):
    """
    Fetches pools without any market cap limit.
    In 'hybrid' mode, combines both Trending Pools and New Pools so low-cap gems
    and established runners are discovered together.
    """
    pools = []
    seen_keys = set()

    if scan_mode == "hybrid":
        feeds = ["new_pools", "trending_pools"]
    elif scan_mode == "low_cap_focus":
        feeds = ["new_pools"]
    else:
        feeds = ["trending_pools"]

    for feed_type in feeds:
        endpoint = (
            f"{BASE_URL}/networks/{feed_type}"
            if selected_network == "all"
            else f"{BASE_URL}/networks/{selected_network}/{feed_type}"
        )

        for page in range(1, pages + 1):
            try:
                resp = session.get(endpoint, params={"page": page}, timeout=15)
                if resp.status_code == 429:
                    time.sleep(10)
                    resp = session.get(endpoint, params={"page": page}, timeout=15)
                resp.raise_for_status()
                data = resp.json().get("data", [])

                for item in data:
                    attr = item.get("attributes", {})
                    rel = item.get("relationships", {})

                    net_id = rel.get("network", {}).get("data", {}).get("id", selected_network)
                    if net_id == "all":
                        pool_id = item.get("id", "")
                        net_id = pool_id.split("_")[0] if "_" in pool_id else "unknown"

                    base_id = rel.get("base_token", {}).get("data", {}).get("id", "")
                    ca = base_id.split("_", 1)[1] if "_" in base_id else base_id

                    unique_key = f"{net_id}:{ca.lower()}"
                    if not ca or ca in NATIVE_WRAPPED_CAS or ca.lower() in NATIVE_WRAPPED_CAS or unique_key in seen_keys:
                        continue

                    reserve = float(attr.get("reserve_in_usd") or 0)
                    vol_24h = float(attr.get("volume_usd", {}).get("h24") or 0)
                    price_change_24h = float(attr.get("price_change_percentage", {}).get("h24") or 0)
                    mcap = float(attr.get("market_cap_usd") or attr.get("fdv_usd") or 0)
                    name = attr.get("name", "UNKNOWN").split("/")[0].strip()
                    pool_address = attr.get("address")

                    # In low-cap focus mode, prioritize tokens under $25M MC/FDV
                    if scan_mode == "low_cap_focus" and mcap > 25_000_000:
                        continue

                    if reserve >= min_liq and vol_24h >= min_vol and pool_address:
                        seen_keys.add(unique_key)
                        pools.append({
                            "symbol": name,
                            "network": net_id,
                            "ca": ca,
                            "pool_address": pool_address,
                            "liquidity": reserve,
                            "vol_24h": vol_24h,
                            "change_24h": price_change_24h,
                            "mcap": mcap,
                            "source_feed": feed_type,
                        })
                time.sleep(1.1)
            except Exception:
                continue

    # Sort so smaller market cap coins are evaluated first
    pools.sort(key=lambda x: (x["mcap"] if x["mcap"] > 0 else 1_000_000))
    return pools


def fetch_pool_candles(session, network, pool_address, timeframe="day", aggregate=1, limit=30):
    for _ in range(3):
        try:
            resp = session.get(
                f"{BASE_URL}/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}",
                params={"aggregate": aggregate, "limit": limit, "currency": "usd"},
                timeout=15,
            )
            if resp.status_code == 429:
                time.sleep(10)
                continue
            resp.raise_for_status()
            ohlcv_list = resp.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])
            if not ohlcv_list:
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv_list,
                columns=["timestamp", "open", "high", "low", "close", "quote_volume"],
            )
            for col in ["open", "high", "low", "close", "quote_volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.dropna().sort_values("timestamp").reset_index(drop=True)
        except Exception:
            time.sleep(2)
    return pd.DataFrame()


def fetch_live_pool_price(session, network, pool_address):
    try:
        resp = session.get(f"{BASE_URL}/networks/{network}/pools/{pool_address}", timeout=10)
        if resp.status_code == 200:
            attr = resp.json().get("data", {}).get("attributes", {})
            return float(attr.get("base_token_price_usd") or attr.get("price_in_usd") or 0)
    except Exception:
        pass
    return None


# =========================
# TELEGRAM COMMAND LISTENER
# =========================
def handle_telegram_commands(bot_token, chat_id, min_vol_ratio, min_liquidity, min_daily_vol, boost_low_cap):
    if not bot_token or not chat_id:
        return "Missing Bot Token or Chat ID."
    
    url = f"https://api.telegram.org/bot{bot_token.strip()}/getUpdates"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return f"Telegram API Error: HTTP {resp.status_code}"
        
        updates = resp.json().get("result", [])
        if not updates:
            return "No new incoming Telegram commands found."
        
        session = create_session()
        commands_processed = 0

        for update in updates:
            msg = update.get("message", {})
            text = msg.get("text", "").strip().lower()
            sender_chat_id = str(msg.get("chat", {}).get("id", ""))

            if sender_chat_id == str(chat_id).strip():
                if "/status" in text:
                    send_telegram_message(
                        bot_token, chat_id,
                        "🟢 <b>NimiQuant Crypto Intelligence:</b>\nOnline, scanning both Trending & New Pools for low-cap and macro breakouts."
                    )
                    commands_processed += 1
                elif "/scan" in text:
                    send_telegram_message(
                        bot_token, chat_id,
                        "🔍 <b>Remote Hybrid Scan Triggered!</b>\nHunting low-cap gems and active breakouts..."
                    )
                    pools = fetch_hybrid_pools(
                        session, "all", scan_mode="hybrid", pages=2,
                        min_liq=min_liquidity, min_vol=min_daily_vol
                    )
                    found = False
                    for p in pools[:6]:
                        daily_df = fetch_pool_candles(session, p["network"], p["pool_address"], timeframe="day", limit=15)
                        h4_df = fetch_pool_candles(session, p["network"], p["pool_address"], timeframe="hour", aggregate=4, limit=12)
                        signal = evaluate_trade_setup(p, daily_df, h4_df, min_vol_ratio, boost_low_cap)
                        if signal:
                            send_telegram_message(bot_token, chat_id, format_trade_alert(signal))
                            found = True
                            break
                    if not found:
                        send_telegram_message(
                            bot_token, chat_id,
                            "⚠️ Remote scan complete: No high-conviction setups cleared threshold right now."
                        )
                    commands_processed += 1

        return f"Successfully processed {commands_processed} Telegram command(s)!"
    except Exception as e:
        return f"Error checking commands: {str(e)}"


# =========================
# MULTI-TIER QUANT ENGINE
# =========================
def classify_market_cap(mcap):
    if mcap <= 0:
        return "💎 Micro-Cap (Early)"
    elif mcap < 5_000_000:
        return "💎 Micro-Cap (<$5M)"
    elif mcap < 50_000_000:
        return "🌱 Low-Cap (<$50M)"
    elif mcap < 250_000_000:
        return "⚡ Mid-Cap (<$250M)"
    else:
        return "🏛️ Large-Cap ($250M+)"


def evaluate_trade_setup(pool, daily_df, h4_df, min_vol_ratio, boost_low_cap=True):
    tier = None
    priority = 0
    vol_ratio = 0.0
    breakout_pct = 0.0
    close_price = 0.0

    # Daily breakout checks (for established & mid-age tokens)
    if len(daily_df) >= 8:
        today = daily_df.iloc[-1]
        close_price = today["close"]
        vol_today = today["quote_volume"]

        if len(daily_df) >= 21:
            hist_20 = daily_df.iloc[-21:-1]
            high_20 = hist_20["high"].max()
            avg_vol_20 = hist_20["quote_volume"].mean()
            vr_20 = (vol_today / avg_vol_20) if avg_vol_20 > 0 else 0

            if close_price > high_20 and vr_20 >= min_vol_ratio:
                tier = "🔥 TIER 1: 20-Day Macro Breakout"
                priority = 3
                vol_ratio = vr_20
                breakout_pct = ((close_price - high_20) / high_20) * 100

        if tier is None:
            hist_7 = daily_df.iloc[-8:-1]
            high_7 = hist_7["high"].max()
            avg_vol_7 = hist_7["quote_volume"].mean()
            vr_7 = (vol_today / avg_vol_7) if avg_vol_7 > 0 else 0

            if close_price > high_7 and vr_7 >= min_vol_ratio:
                tier = "⚡ TIER 2: 7-Day Momentum Runner"
                priority = 2
                vol_ratio = vr_7
                breakout_pct = ((close_price - high_7) / high_7) * 100

    # 4-Hour Fast-Track check (catches younger low-cap coins with < 8 days of history!)
    if tier is None and len(h4_df) >= 8:
        curr_4h = h4_df.iloc[-1]
        close_price = curr_4h["close"]
        vol_4h = curr_4h["quote_volume"]

        lookback = min(13, len(h4_df))
        hist_4h = h4_df.iloc[-lookback:-1]
        high_recent = hist_4h["high"].max()
        avg_vol_4h = hist_4h["quote_volume"].mean()
        vr_4h = (vol_4h / avg_vol_4h) if avg_vol_4h > 0 else 0

        if close_price > high_recent and vr_4h >= min_vol_ratio:
            tier = "🎯 TIER 3: 4-Hour Active Breakout"
            priority = 1
            vol_ratio = vr_4h
            breakout_pct = ((close_price - high_recent) / high_recent) * 100

    if tier is None:
        return None

    mcap = pool.get("mcap", 0)
    cap_tier = classify_market_cap(mcap)

    # Low-Cap Conviction Bonus so smaller gems rank higher alongside big coins
    low_cap_bonus = 0
    if boost_low_cap:
        if 0 < mcap < 10_000_000 or mcap == 0:
            low_cap_bonus = 15
        elif mcap < 50_000_000:
            low_cap_bonus = 8

    score = min(100, round((priority * 14) + (vol_ratio * 17) + (breakout_pct * 2.5) + low_cap_bonus, 1))

    if priority == 3:
        stop_loss = round(close_price * 0.90, 6)
        take_profit = round(close_price * 1.35, 6)
    elif priority == 2:
        stop_loss = round(close_price * 0.92, 6)
        take_profit = round(close_price * 1.25, 6)
    else:
        stop_loss = round(close_price * 0.94, 6)
        take_profit = round(close_price * 1.18, 6)

    net_label, wallets = NETWORK_WALLET_GUIDE.get(
        pool["network"],
        (pool["network"].upper(), "Any Web3 Wallet supporting this network"),
    )

    return {
        "Tier": tier,
        "Cap Tier": cap_tier,
        "Priority": priority,
        "Symbol": pool["symbol"],
        "Network": net_label,
        "Raw Network": pool["network"],
        "Pool Address": pool["pool_address"],
        "Compatible Wallets": wallets,
        "Score": score,
        "Market Cap ($)": round(mcap, 0),
        "Price ($)": close_price,
        "24h Change (%)": round(pool["change_24h"], 2),
        "Volume Surge": round(vol_ratio, 2),
        "24h Volume ($)": round(pool["vol_24h"], 0),
        "Liquidity ($)": round(pool["liquidity"], 0),
        "Stop-Loss": stop_loss,
        "Take-Profit": take_profit,
        "Contract Address": pool["ca"],
    }


# =========================
# STREAMLIT APP UI
# =========================
st.set_page_config(page_title="NimiQuant Crypto Intelligence Engine", page_icon="⚡", layout="wide")

if "sent_alerts" not in st.session_state:
    st.session_state.sent_alerts = set()

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# --- STYLISH INSTITUTIONAL HOME PAGE HEADER ---
st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h1 style="margin-bottom: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; color: #FFFFFF;">
            NimiQuant Crypto Intelligence
        </h1>
        <p style="margin-top: 4px; font-size: 1rem; font-weight: 600; color: #38BDF8; letter-spacing: 0.3px;">
            ⚡ Live Trade Scanner & Portfolio PnL Tracker
        </p>
    </div>
""", unsafe_allow_html=True)

st.caption("High-Precision Dual-Timeframe Engine (20D / 7D / 4H) — Multi-Chain & Low-Cap Gem Discovery")

with st.sidebar:
    st.header("📢 Telegram Integration")
    enable_telegram = st.toggle("Enable Instant Phone Alerts", value=bool(DEFAULT_BOT_TOKEN and DEFAULT_CHAT_ID))
    
    tg_token = DEFAULT_BOT_TOKEN
    tg_chat_id = DEFAULT_CHAT_ID
    
    min_alert_score = st.slider("Min Conviction Score to Alert", 40, 95, 55, 5)

    if st.button("🔔 Send Test Telegram Alert", use_container_width=True):
        ok, info = send_telegram_message(
            tg_token,
            tg_chat_id,
            "✅ <b>NimiQuant Crypto Intelligence Connected!</b>\nYour live multi-chain & low-cap breakout alerts are active.",
        )
        if ok:
            st.success("Test alert delivered to your Telegram!")
        else:
            st.error(f"Telegram Error: {info}")

    st.divider()
    st.header("⚙️ Strategy & Gem Sensitivity")
    boost_low_cap = st.toggle(
        "Prioritize Low-Cap Gems (<$50M)",
        value=True,
        help="Gives low-cap and micro-cap coins a conviction score boost so they rank alongside large-cap coins without setting a hard limit."
    )
    min_vol_ratio = st.slider("Min Volume Surge Multiplier", 1.1, 2.5, 1.3, 0.1)
    min_liquidity = st.number_input("Min Pool Liquidity ($)", 10_000, 1_000_000, 25_000, 5_000)
    min_daily_vol = st.number_input("Min 24h Volume ($)", 20_000, 2_000_000, 50_000, 10_000)
    pages_to_scan = st.selectbox("Market Depth (Pages per Feed)", [2, 3, 4], index=1)

    if st.button("📥 Check Telegram Commands (/scan)", use_container_width=True):
        with st.spinner("Checking Telegram for /scan or /status commands..."):
            res_msg = handle_telegram_commands(tg_token, tg_chat_id, min_vol_ratio, min_liquidity, min_daily_vol, boost_low_cap)
            st.info(res_msg)

# =========================
# MULTI-TAB INTERFACE
# =========================
tab_scanner, tab_portfolio = st.tabs(["⚡ Market Scanner & Signals", "💼 Portfolio & Live PnL Tracker"])

with tab_scanner:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Discovery Mode", "Hybrid (All Caps)")
    col2.metric("Low-Cap Boost", "🟢 Active" if boost_low_cap else "⚪ Off")
    col3.metric("Volume Trigger", f"≥ {min_vol_ratio}x Avg")
    col4.metric("Telegram Status", "🟢 Live" if (enable_telegram and tg_token and tg_chat_id) else "⚪ Standby")

    st.divider()

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        network_options = {
            "🌐 All Blockchains (Global Scan)": "all",
            "🟣 Solana (Phantom / Solflare / Trust)": "solana",
            "🔵 Base (Coinbase / MetaMask / Trust)": "base",
            "💎 Ethereum (MetaMask / Trust / Rabby)": "eth",
            "🟡 BNB Chain / BSC (Trust / MetaMask)": "bsc",
            "🔷 Arbitrum (MetaMask / Rabby / Trust)": "arbitrum",
            "🟪 Polygon (MetaMask / Trust / Phantom)": "polygon_pos",
        }
        selected_label = st.selectbox("Select Target Blockchain Network:", list(network_options.keys()))
        selected_network = network_options[selected_label]

    with filter_col2:
        mode_options = {
            "⚡ Hybrid Scan (Trending + Low-Cap Gems)": "hybrid",
            "💎 Pure Low-Cap Hunter (New & Under $25M)": "low_cap_focus",
            "🏛️ Trending Runners Only (All Caps)": "trending_only",
        }
        selected_mode_label = st.selectbox("Select Discovery Strategy:", list(mode_options.keys()))
        selected_mode = mode_options[selected_mode_label]

    if st.button("🚀 SCAN GLOBAL MARKET FOR ACTIVE TRADES", type="primary", use_container_width=True):
        session = create_session()
        progress_bar = st.progress(0, text="Scanning Trending & New Pools across networks...")
        pools = fetch_hybrid_pools(
            session=session,
            selected_network=selected_network,
            scan_mode=selected_mode,
            pages=pages_to_scan,
            min_liq=min_liquidity,
            min_vol=min_daily_vol,
        )

        if not pools:
            progress_bar.empty()
            st.warning("No pools matched your minimum liquidity/volume filters. Try lowering Min Pool Liquidity.")
        else:
            results = []
            alerts_sent = 0
            total = len(pools)

            for idx, pool in enumerate(pools, start=1):
                progress_bar.progress(
                    idx / total,
                    text=f"High-Precision Scan [{idx}/{total}]: {pool['symbol']} ({pool['network'].upper()})...",
                )
                daily_df = fetch_pool_candles(session, pool["network"], pool["pool_address"], timeframe="day", aggregate=1, limit=25)
                time.sleep(1.8)
                h4_df = fetch_pool_candles(session, pool["network"], pool["pool_address"], timeframe="hour", aggregate=4, limit=20)
                time.sleep(1.8)

                signal = evaluate_trade_setup(pool, daily_df, h4_df, min_vol_ratio, boost_low_cap)
                if signal:
                    results.append(signal)

                    alert_key = f"{signal['Network']}:{signal['Contract Address']}:{signal['Tier']}"
                    if (
                        enable_telegram
                        and signal["Score"] >= min_alert_score
                        and alert_key not in st.session_state.sent_alerts
                    ):
                        ok, _ = send_telegram_message(tg_token, tg_chat_id, format_trade_alert(signal))
                        if ok:
                            st.session_state.sent_alerts.add(alert_key)
                            alerts_sent += 1

            progress_bar.empty()

            if not results:
                st.warning("No active 20D, 7D, or 4H breakouts triggered right now. Try lowering the Volume Surge slider to 1.1x.")
            else:
                res_df = pd.DataFrame(results).sort_values(["Score", "Priority"], ascending=[False, False]).reset_index(drop=True)
                alert_note = f" | 📲 {alerts_sent} Telegram alert(s) sent!" if enable_telegram else ""
                st.success(f"Scan Complete! Found {len(res_df)} Active Trade Setups{alert_note}")

                for _, row in res_df.iterrows():
                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns([2.4, 1.2, 1.2, 1.2])
                        c1.subheader(f"{row['Symbol']}  `[{row['Network']}]`")
                        c1.caption(f"**{row['Tier']}**  •  {row['Cap Tier']}")
                        c2.metric("Conviction Score", f"{row['Score']} / 100")
                        c3.metric("Entry Price", f"${row['Price ($)']:.6f}", f"{row['24h Change (%)']}%")
                        c4.metric("Volume Surge", f"{row['Volume Surge']}x avg")

                        mcap_display = f"${row['Market Cap ($)']:,.0f}" if row["Market Cap ($)"] > 0 else "Unverified / Early Gem"
                        st.markdown(f"**Market Cap / FDV:** `{mcap_display}`  |  **Supported Wallets:** {row['Compatible Wallets']}")
                        st.markdown(f"**Trade Plan:** Entry: `${row['Price ($)']:.6f}`  |  Stop-Loss: `${row['Stop-Loss']}`  |  Take-Profit Target: `${row['Take-Profit']}`")
                        st.caption(f"24h Volume: ${row['24h Volume ($)']:,.0f}  |  Pool Liquidity: ${row['Liquidity ($)']:,.0f}")
                        st.code(row["Contract Address"], language=None)

                        if st.button(f"📥 Log {row['Symbol']} to Portfolio", key=f"log_{row['Pool Address']}"):
                            st.session_state.portfolio.append({
                                "Symbol": row["Symbol"],
                                "Network": row["Network"],
                                "Raw Network": row["Raw Network"],
                                "Pool Address": row["Pool Address"],
                                "Entry Price": row["Price ($)"],
                                "Current Price": row["Price ($)"],
                                "Stop-Loss": row["Stop-Loss"],
                                "Take-Profit": row["Take-Profit"],
                                "PnL (%)": 0.0,
                                "Status": "Active 🟢",
                                "Contract Address": row["Contract Address"]
                            })
                            st.success(f"Successfully logged {row['Symbol']} to your Live PnL Tracker!")

with tab_portfolio:
    st.subheader("💼 Active Positions & Live PnL Tracker")
    st.caption("Live price polling, percentage PnL calculation, and risk-management target tracking.")

    if not st.session_state.portfolio:
        st.info("No trades logged yet. Scan the market in the first tab and click 'Log to Portfolio' on any trade setup!")
    else:
        session = create_session()
        if st.button("🔄 Refresh Live Prices & Calculate PnL", type="primary"):
            with st.spinner("Polling GeckoTerminal for live token prices..."):
                for trade in st.session_state.portfolio:
                    if "Active" in trade["Status"]:
                        live_p = fetch_live_pool_price(session, trade["Raw Network"], trade["Pool Address"])
                        if live_p and live_p > 0:
                            entry_p = trade["Entry Price"]
                            trade["Current Price"] = live_p
                            pnl = ((live_p - entry_p) / entry_p) * 100
                            trade["PnL (%)"] = round(pnl, 2)
                            
                            if live_p >= trade["Take-Profit"]:
                                trade["Status"] = "Target Hit 🎯 (+)"
                            elif live_p <= trade["Stop-Loss"]:
                                trade["Status"] = "Stop-Loss Hit 🛑 (-)"
            st.success("Live prices successfully updated!")

        port_df = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(port_df, use_container_width=True)

        # CSV Export Button
        csv_data = port_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Trade History to CSV",
            data=csv_data,
            file_name="nimiquant_trade_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear Portfolio Log", type="secondary"):
            st.session_state.portfolio = []
            st.rerun()
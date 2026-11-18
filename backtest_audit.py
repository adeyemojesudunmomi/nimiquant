
import pandas as pd
import portfolio_backtest as pb

# NimiQuant — Backtest Audit
# Compares momentum against BTC and tests higher trading costs.

def main():
    print("\nNimiQuant — Backtest Audit")
    print("=" * 42)

    df = pb.calculate_signals(pb.load_data())

    # Match the original out-of-sample test period.
    dates = sorted(pd.Timestamp(x) for x in df["date"].unique())
    split = int(len(dates) * pb.TRAINING_FRACTION)
    test_dates = dates[split:]

    print(f"Test period: {test_dates[0].date()} to {test_dates[-1].date()}")
    print(f"Starting capital: ${pb.STARTING_CAPITAL:,.2f}")

    # BTC buy-and-hold benchmark, including entry and exit fees.
    btc = df[df["symbol"] == "BTCUSDT"].set_index("date").sort_index()

    first_date = test_dates[0]
    last_date = test_dates[-1]

    if first_date in btc.index and last_date in btc.index:
        entry = float(btc.loc[first_date, "open"])
        exit_price = float(btc.loc[last_date, "close"])

        btc_return = (
            exit_price * (1 - 0.0015)
            / (entry * (1 + 0.0015)) - 1
        ) * 100

        btc_final = pb.STARTING_CAPITAL * (1 + btc_return / 100)

        print("\nBTC BUY-AND-HOLD")
        print(f"Final value: ${btc_final:,.2f}")
        print(f"Total return: {btc_return:.2f}%")
    else:
        print("\nBTC benchmark unavailable for exact test dates.")

    # Re-run the strategy with increasingly expensive trading.
    original_cost = pb.COST_PER_SIDE
    results = []

    try:
        for cost in [0.0015, 0.0030, 0.0050]:
            pb.COST_PER_SIDE = cost

            result = pb.simulate_portfolio(
                df, test_dates, None, "momentum"
            )

            results.append({
                "cost_per_side_pct": cost * 100,
                "final_equity": result["final_equity"],
                "total_return_pct": result["total_return_pct"],
                "max_drawdown_pct": result["max_drawdown_pct"],
                "trades": result["trades"],
            })
    finally:
        pb.COST_PER_SIDE = original_cost

    results_df = pd.DataFrame(results)

    print("\nMOMENTUM — COST SENSITIVITY")
    print(results_df.round(2).to_string(index=False))

    output = pb.OUTPUT_DIR / "backtest_audit_results.csv"
    results_df.to_csv(output, index=False)

    print(f"\nSaved: {output}")
    print("\nRemember: this audit is not proof of future profitability.")


if __name__ == "__main__":
    main()

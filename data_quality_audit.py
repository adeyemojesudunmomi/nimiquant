
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "prices.csv"


def main():
    print("\nNimiQuant — Historical Data Quality Audit")
    print("=" * 48)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "data/prices.csv not found. Run py data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = {"date", "symbol", "open", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(
        df["date"], utc=True, errors="coerce"
    )

    df = df.dropna(subset=["date", "symbol"])
    df = df.sort_values(["symbol", "date"])

    print(f"Rows: {len(df):,}")
    print(f"Coins: {df['symbol'].nunique()}")
    print(f"Unique dates: {df['date'].nunique()}")

    duplicate_count = int(
        df.duplicated(["symbol", "date"]).sum()
    )
    print(f"Duplicate coin/date rows: {duplicate_count}")

    dates_per_coin = df.groupby("symbol")["date"].agg(
        first_date="min",
        last_date="max",
        observed_days="nunique",
    )

    total_dates = df["date"].nunique()

    dates_per_coin["missing_days_in_span"] = (
        (
            dates_per_coin["last_date"]
            - dates_per_coin["first_date"]
        ).dt.days + 1
        - dates_per_coin["observed_days"]
    )

    dates_per_coin["coverage_pct"] = (
        dates_per_coin["observed_days"]
        / total_dates * 100
    )

    print("\nCoins with the most missing dates within their span:")
    print(
        dates_per_coin.sort_values(
            "missing_days_in_span", ascending=False
        ).head(15).round(2).to_string()
    )

    print("\nPrice validity:")
    for column in ["open", "close"]:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = int(
            (values.isna() | (values <= 0)).sum()
        )
        print(f"Invalid {column} values: {invalid}")

    # Save detailed results for later investigation.
    output_file = (
        BASE_DIR / "data" / "data_quality_audit.csv"
    )

    dates_per_coin.to_csv(output_file)

    print(f"\nDetailed report saved to: {output_file}")
    print(
        "\nNOTE: Missing dates are diagnostic flags, not "
        "proof of bad data. Newly listed coins and delistings "
        "can naturally have shorter histories."
    )


if __name__ == "__main__":
    main()

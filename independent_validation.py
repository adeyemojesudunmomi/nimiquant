
from pathlib import Path
import ast

PROJECT_DIR = Path(__file__).resolve().parent
BACKTEST_FILE = PROJECT_DIR / "portfolio_backtest.py"
DATA_FILE = PROJECT_DIR / "data" / "prices.csv"
RESULTS_FILE = PROJECT_DIR / "data" / "portfolio_backtest_results.csv"

print("=" * 55)
print("NimiQuant — Independent Validation: Check 1")
print("=" * 55)

# 1. Check required files
for file_path in [BACKTEST_FILE, DATA_FILE, RESULTS_FILE]:
    status = "FOUND" if file_path.exists() else "MISSING"
    print(f"{status}: {file_path.name}")

if not BACKTEST_FILE.exists():
    raise SystemExit("\nCannot continue: portfolio_backtest.py is missing.")

# 2. Check that the backtest code is valid Python
source = BACKTEST_FILE.read_text(encoding="utf-8")

try:
    tree = ast.parse(source)
    print("\nPASS: Backtest code parses successfully.")
except SyntaxError as error:
    raise SystemExit(f"\nFAIL: Syntax error: {error}")

# 3. Inspect important simulation functions
functions = {
    node.name: node
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}

print("\nSimulation functions:")
for name in ["calculate_signals", "simulate_portfolio", "load_data"]:
    if name in functions:
        print(f"FOUND: {name}()")
    else:
        print(f"WARNING: {name}() not found")

if "simulate_portfolio" in functions:
    node = functions["simulate_portfolio"]
    function_source = ast.get_source_segment(source, node) or ""

    checks = {
        "Uses entry open prices": "open" in function_source.lower(),
        "Uses exit/closing prices": "close" in function_source.lower(),
        "Includes transaction-cost logic": (
            "COST_PER_SIDE" in function_source
            or "cost_per_side" in function_source
        ),
        "Tracks equity over time": (
            "equity_records" in function_source
            or "equity_curve" in function_source
        ),
        "Has position exit logic": (
            "exit_index" in function_source
            or "exit_date" in function_source
        ),
    }

    print("\nCode inspection:")
    for description, passed in checks.items():
        print(f"{'FOUND' if passed else 'REVIEW'}: {description}")

# 4. Check historical data quality
if DATA_FILE.exists():
    import pandas as pd

    data = pd.read_csv(DATA_FILE)
    required = {"date", "symbol", "open", "close", "quote_volume"}
    missing = required - set(data.columns)

    print("\nHistorical data checks:")
    print(f"Rows: {len(data):,}")
    print(f"Coins: {data['symbol'].nunique() if 'symbol' in data else 'UNKNOWN'}")
    print(f"Dates: {data['date'].nunique() if 'date' in data else 'UNKNOWN'}")

    if missing:
        print(f"FAIL: Missing columns: {sorted(missing)}")
    else:
        print("PASS: Required price columns exist.")

        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        for col in ["open", "close", "quote_volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        print(f"Duplicate coin/date rows: {data.duplicated(['date', 'symbol']).sum():,}")
        print(f"Invalid dates: {data['date'].isna().sum():,}")
        print(f"Invalid/non-positive prices: {((data['open'] <= 0) | (data['close'] <= 0)).sum():,}")
        print(f"Missing price/volume values: {data[['open', 'close', 'quote_volume']].isna().any(axis=1).sum():,}")

# 5. Display existing results without treating them as proof
if RESULTS_FILE.exists():
    print("\nExisting backtest results:")
    print(pd.read_csv(RESULTS_FILE).to_string(index=False))
else:
    print("\nNo saved portfolio results file found.")

print("\n" + "=" * 55)
print("CHECK COMPLETE")
print("This is a diagnostic, not proof of profitability.")
print("=" * 55)

# main.py
import argparse, json
from utils import *

def parse_args():
    ap = argparse.ArgumentParser(description="Section 4 JSON export (matches notebook).")
    ap.add_argument("--csv", required=True, help="Path to RAW transactions CSV (with date/category/amount/description...).")
    ap.add_argument("--out", required=True, help="Path to write JSON.")
    return ap.parse_args()

def main():
    args = parse_args()

    # 1) read raw
    df_raw = read_df(args.csv)

    # 2) compute df_tx (flags + refunds + effective_outflow)
    dftx = df_tx(df_raw)

    # 3) latest month anomalies (exact same filter as notebook)
    anoms, last_month_str = latest_month_anomalies(dftx, df_raw=df_raw)

    # 4) build JSON payload and save
    payload = build_payload(anoms)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Last month: {last_month_str} | anomalies: {len(payload['value'])}")
    print(f"Saved JSON to: {args.out}")

if __name__ == "__main__":
    main()

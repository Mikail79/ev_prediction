"""
fetch_charging_data.py — Fetch EV Charging Station Data from AFDC API
=====================================================================
Downloads all electric vehicle charging stations in Washington State
from the U.S. DOE Alternative Fuels Data Center (AFDC) API,
then builds a monthly cumulative time-series of station/port counts.

Outputs:
  - charging_stations.csv (monthly cumulative counts)

Usage:
  python fetch_charging_data.py
"""

import json
import os
import urllib.request
from datetime import datetime

import pandas as pd

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
API_KEY = "DEMO_KEY"
BASE_URL = "https://developer.nlr.gov/api/alt-fuel-stations/v1.json"
STATE = "WA"
FUEL_TYPE = "ELEC"
PAGE_SIZE = 200

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.path.abspath("")

OUTPUT_PATH = os.path.join(base_dir, "charging_stations.csv")


def fetch_all_stations():
    """Fetch all electric charging stations in WA via paginated API calls."""
    all_stations = []
    offset = 0

    print("=" * 60)
    print("  AFDC Charging Station Data Fetcher")
    print("=" * 60)

    while True:
        url = (
            f"{BASE_URL}?api_key={API_KEY}"
            f"&fuel_type={FUEL_TYPE}&state={STATE}"
            f"&status=E&limit={PAGE_SIZE}&offset={offset}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        total = data["total_results"]
        stations = data["fuel_stations"]
        all_stations.extend(stations)

        print(f"   [PAGE] Fetched {len(all_stations)}/{total} stations...")

        if len(all_stations) >= total:
            break
        offset += PAGE_SIZE

    print(f"   [OK] Total stations fetched: {len(all_stations)}")
    return all_stations, total


def build_monthly_timeseries(stations):
    """
    Build a monthly cumulative time-series from station open_dates.
    For each month, count how many stations had opened by that date,
    and sum the cumulative number of Level 2 and DC Fast ports.
    """
    records = []
    for s in stations:
        open_date = s.get("open_date")
        if not open_date:
            continue
        records.append({
            "open_date": pd.to_datetime(open_date),
            "ports_l2": s.get("ev_level2_evse_num") or 0,
            "ports_dcfc": s.get("ev_dc_fast_num") or 0,
        })

    df = pd.DataFrame(records)
    print(f"   [DATA] Stations with valid open_date: {len(df)}")

    # Create month-end periods for grouping
    df["month"] = df["open_date"].dt.to_period("M")

    # Count new stations per month and sum new ports
    monthly = df.groupby("month").agg(
        new_stations=("open_date", "count"),
        new_ports_l2=("ports_l2", "sum"),
        new_ports_dcfc=("ports_dcfc", "sum"),
    ).sort_index()

    # Build a complete monthly range (no gaps)
    full_range = pd.period_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="M",
    )
    monthly = monthly.reindex(full_range, fill_value=0)

    # Calculate cumulative sums
    result = pd.DataFrame({
        "Date": monthly.index.to_timestamp(how="end"),
        "Stations_Cumulative": monthly["new_stations"].cumsum().values,
        "Ports_L2_Cumulative": monthly["new_ports_l2"].cumsum().values,
        "Ports_DCFC_Cumulative": monthly["new_ports_dcfc"].cumsum().values,
    })

    return result


def main():
    try:
        stations, total = fetch_all_stations()
        ts = build_monthly_timeseries(stations)

        # Save to CSV
        ts.to_csv(OUTPUT_PATH, index=False)

        print(f"\n[SAVE] Saved to {OUTPUT_PATH}")
        print(f"   Rows: {len(ts)}")
        print(f"   Date range: {ts['Date'].min().strftime('%b %Y')} -> {ts['Date'].max().strftime('%b %Y')}")
        print(f"   Latest stations: {ts['Stations_Cumulative'].iloc[-1]:,}")
        print(f"   Latest L2 ports: {ts['Ports_L2_Cumulative'].iloc[-1]:,}")
        print(f"   Latest DCFC ports: {ts['Ports_DCFC_Cumulative'].iloc[-1]:,}")

    except Exception as e:
        print(f"\n[ERROR] Failed to fetch from API: {e}")
        if os.path.exists(OUTPUT_PATH):
            print(f"   [FALLBACK] Using existing local file: {OUTPUT_PATH}")
        else:
            raise RuntimeError("No charging station data available.") from e


if __name__ == "__main__":
    main()

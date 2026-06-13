# data/qualifying.py
import requests
import pandas as pd
from config import JOLPICA_BASE


def get_qualifying_results(season: int, round_number: int) -> pd.DataFrame | None:
    """
    Fetch qualifying results from Jolpica for a given season/round.

    Returns:
        DataFrame with columns:
            - driver_code
            - constructor_id
            - grid_position    (integer 1..N)
            - best_quali_time  (seconds, best of Q3/Q2/Q1 available)
            - quali_gap_to_pole (seconds delta to pole sitter; 0.0 for pole)
            - penalty_flag     (1 if grid==0 i.e. pit lane start)
        Returns None on failure.
    """
    url = f"{JOLPICA_BASE}/{season}/{round_number}/qualifying.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        races = resp.json()["MRData"]["RaceTable"]["Races"]
    except Exception as e:
        print(f"[WARNING] [qualifying] Failed to fetch {season} round {round_number}: {e}")
        return None

    if not races or "QualifyingResults" not in races[0]:
        print(f"[WARNING] [qualifying] No qualifying data for {season} round {round_number}")
        return None

    quali_results = races[0]["QualifyingResults"]
    rows = []
    for r in quali_results:
        best_time = _best_quali_time(r)
        rows.append({
            "driver_code":    r["Driver"]["code"],
            "constructor_id": r["Constructor"]["constructorId"],
            "grid_position":  int(r["position"]),
            "best_quali_time": best_time,
            "penalty_flag":   0,  # grid penalties resolved by race grid, not quali position
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # Compute gap to pole (pole sitter has best_quali_time and gap = 0)
    pole_time = df["best_quali_time"].min()
    df["quali_gap_to_pole"] = df["best_quali_time"] - pole_time
    # Pole sitter should be exactly 0
    df.loc[df["best_quali_time"] == pole_time, "quali_gap_to_pole"] = 0.0

    return df


def _parse_lap_time(time_str: str) -> float | None:
    """Convert 'm:ss.sss' or 'ss.sss' lap time string to float seconds."""
    if not time_str:
        return None
    try:
        if ":" in time_str:
            parts = time_str.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except (ValueError, IndexError):
        return None


def _best_quali_time(result: dict) -> float:
    """
    Return the driver's best qualifying time in seconds.
    Preference: Q3 > Q2 > Q1. If none available, return a large penalty value.
    """
    for q_key in ("Q3", "Q2", "Q1"):
        t = _parse_lap_time(result.get(q_key, ""))
        if t is not None:
            return t
    return 999.0  # No time recorded — relegated to back

# data/stints.py
import pandas as pd
import numpy as np
from config import OPENF1_BASE, FALLBACK_COMPOUND_DEGRADATION, OPENF1_MIN_YEAR, requests_get_with_retry


def get_compound_degradation(session_key: int, use_fastf1_laps: pd.DataFrame = None) -> float:
    """
    Compute compound degradation rate (seconds/lap) for the primary compound
    using OpenF1 stints + lap time data.

    Strategy:
        1. Fetch stints from OpenF1 to get stint boundaries per driver
        2. For the longest stint on the primary compound, compute
           degradation = (median last-3-lap time - median first-3-lap time) / stint_length
        3. Average across all drivers on the primary compound

    Args:
        session_key: OpenF1 session key for the session
        use_fastf1_laps: Optional FastF1 laps DataFrame to use for lap times
                         (more accurate than OpenF1 laps for FP sessions)

    Returns:
        float: seconds per lap degradation rate (positive = getting slower)
    """
    stints = _fetch_stints(session_key)
    if stints is None or stints.empty:
        return FALLBACK_COMPOUND_DEGRADATION

    # Determine primary compound (most common across all drivers)
    primary_compound = stints["compound"].mode().iloc[0] if not stints.empty else None
    if primary_compound is None:
        return FALLBACK_COMPOUND_DEGRADATION

    # Filter to primary compound stints with enough laps
    long_stints = stints[
        (stints["compound"] == primary_compound) &
        (stints["stint_length"] >= 8)
    ]

    if long_stints.empty:
        return FALLBACK_COMPOUND_DEGRADATION

    if use_fastf1_laps is not None:
        return _degradation_from_fastf1(long_stints, use_fastf1_laps, primary_compound)
    else:
        return _degradation_from_openf1(session_key, long_stints)


def _fetch_stints(session_key: int) -> pd.DataFrame | None:
    """Fetch all stints for a session from OpenF1."""
    try:
        resp = requests_get_with_retry(
            f"{OPENF1_BASE}/stints",
            params={"session_key": session_key},
            timeout=15,
        )
        data = resp.json()
    except Exception as e:
        print(f"[WARNING] [stints] Failed to fetch stints for session {session_key}: {e}")
        return None

    if not data:
        return None

    df = pd.DataFrame(data)
    df["stint_length"] = df["lap_end"] - df["lap_start"] + 1
    return df


def _degradation_from_openf1(session_key: int, long_stints: pd.DataFrame) -> float:
    """Compute degradation using OpenF1 lap times."""
    # Fetch all laps for this session in a single bulk request to avoid rate limits
    try:
        resp = requests_get_with_retry(
            f"{OPENF1_BASE}/laps",
            params={"session_key": session_key},
            timeout=15,
        )
        laps_data = resp.json()
    except Exception as e:
        print(f"[WARNING] [stints] Bulk laps fetch failed for session {session_key}: {e}")
        return FALLBACK_COMPOUND_DEGRADATION

    if not laps_data:
        return FALLBACK_COMPOUND_DEGRADATION

    laps_df = pd.DataFrame(laps_data)
    if laps_df.empty or "lap_number" not in laps_df.columns or "driver_number" not in laps_df.columns:
        return FALLBACK_COMPOUND_DEGRADATION

    degradation_rates = []

    for _, stint in long_stints.iterrows():
        driver_num = stint["driver_number"]
        lap_start  = stint["lap_start"]
        lap_end    = stint["lap_end"]

        # Filter in-memory instead of making an API request per driver
        stint_laps = laps_df[
            (laps_df["driver_number"] == driver_num) &
            (laps_df["lap_number"] >= lap_start) &
            (laps_df["lap_number"] <= lap_end) &
            laps_df["lap_duration"].notna()
        ].copy()

        if len(stint_laps) < 8:
            continue

        stint_laps = stint_laps.sort_values("lap_number")
        first_3 = stint_laps.head(3)["lap_duration"].median()
        last_3  = stint_laps.tail(3)["lap_duration"].median()
        stint_len = len(stint_laps)

        if first_3 > 0 and stint_len > 0:
            rate = (last_3 - first_3) / stint_len
            if 0 <= rate <= 0.5:  # Sanity check — real degradation is 0-0.5s/lap
                degradation_rates.append(rate)

    if not degradation_rates:
        return FALLBACK_COMPOUND_DEGRADATION

    return float(np.median(degradation_rates))


def _degradation_from_fastf1(long_stints: pd.DataFrame, laps_df: pd.DataFrame, compound: str) -> float:
    """Compute degradation using FastF1 lap times (more accurate)."""
    degradation_rates = []

    compound_laps = laps_df[
        laps_df["Compound"].str.upper() == compound.upper()
    ].copy()

    if compound_laps.empty:
        return FALLBACK_COMPOUND_DEGRADATION

    compound_laps["LapTimeSec"] = compound_laps["LapTime"].dt.total_seconds()
    compound_laps = compound_laps[compound_laps["LapTimeSec"].notna()]

    for driver in compound_laps["Driver"].unique():
        drv_laps = compound_laps[compound_laps["Driver"] == driver].sort_values("LapNumber")
        # Filter out pit in/out laps
        drv_laps = drv_laps[drv_laps["PitOutTime"].isna() & drv_laps["PitInTime"].isna()]
        if len(drv_laps) < 8:
            continue
        first_3 = drv_laps.head(3)["LapTimeSec"].median()
        last_3  = drv_laps.tail(3)["LapTimeSec"].median()
        rate = (last_3 - first_3) / len(drv_laps)
        if 0 <= rate <= 0.5:
            degradation_rates.append(rate)

    if not degradation_rates:
        return FALLBACK_COMPOUND_DEGRADATION

    return float(np.median(degradation_rates))


def get_openf1_session_key(circuit_short_name: str, session_name: str = "Race", year: int = None) -> list[dict]:
    """
    Look up OpenF1 session keys for a given circuit and session type.
    Returns list of session dicts sorted newest-first.
    Only works for 2023+ data.
    """
    if year is not None and year < OPENF1_MIN_YEAR:
        return []

    params = {"circuit_short_name": circuit_short_name, "session_name": session_name}
    if year:
        params["year"] = year

    try:
        resp = requests_get_with_retry(f"{OPENF1_BASE}/sessions", params=params, timeout=10)
        sessions = resp.json()
    except Exception as e:
        print(f"[WARNING] [stints] Session lookup failed for {circuit_short_name}: {e}")
        return []

    return sorted(sessions, key=lambda s: s.get("date_start", ""), reverse=True)

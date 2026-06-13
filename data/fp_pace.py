# data/fp_pace.py
import fastf1
import pandas as pd
import numpy as np
from config import FUEL_CORRECTION_FACTOR, LONG_RUN_MIN_LAPS

fastf1.Cache.enable_cache("cache/")   # local cache to avoid re-downloading


def extract_fp2_pace(season: int, round_number: int) -> pd.DataFrame | None:
    """
    Load FP2, filter for long runs, apply fuel correction, return per-driver median pace.

    Returns:
        DataFrame indexed by driver code with columns:
            - corrected_median_pace   (seconds)
            - qualifying_long_run_laps (int)
            - compound                (most common compound)
            - fp2_pace_delta_to_fastest (seconds delta to fastest driver)
        Returns None if session data is unavailable.
    """
    try:
        session = fastf1.get_session(season, round_number, "FP2")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"[WARNING] [fp_pace] FP2 session unavailable for {season} round {round_number}: {e}")
        return None

    laps = session.laps
    if laps is None or laps.empty:
        print(f"[WARNING] [fp_pace] FP2 laps empty for {season} round {round_number}")
        return None

    return _process_laps(laps)


def _process_laps(laps: pd.DataFrame) -> pd.DataFrame | None:
    """Core processing logic (separated for testability)."""
    laps = laps.copy()

    # 1. Convert LapTime timedelta to float seconds
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    # 2. Exclude pit out / pit in laps
    is_pit_out = laps["PitOutTime"].notna()
    is_pit_in  = laps["PitInTime"].notna()
    laps = laps[~is_pit_out & ~is_pit_in]

    # 3. Exclude deleted laps
    if "Deleted" in laps.columns:
        laps = laps[~laps["Deleted"].fillna(False)]

    # 4. Exclude laps with missing lap time
    laps = laps[laps["LapTimeSec"].notna()]

    if laps.empty:
        return None

    # 5. Exclude outlier laps (>107% of driver's own median — catches SC/VSC laps)
    #    Use transform() to compute per-driver medians aligned to the full DataFrame index,
    #    then filter in-place. Avoids groupby().apply() which drops grouping columns in
    #    newer pandas versions.
    driver_medians = laps.groupby("Driver")["LapTimeSec"].transform("median")
    laps = laps[laps["LapTimeSec"] <= driver_medians * 1.07].copy()

    # 6. Compute stint-relative lap number for fuel correction
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["StintLapNumber"] = laps.groupby(["Driver", "Compound"]).cumcount() + 1

    # 7. Apply fuel correction
    laps["CorrectedLapTime"] = fuel_correct_lap_times(
        laps["LapTimeSec"], laps["StintLapNumber"]
    )

    # 8. Require minimum long-run length per driver
    driver_counts = laps.groupby("Driver")["CorrectedLapTime"].count()
    eligible = driver_counts[driver_counts >= LONG_RUN_MIN_LAPS].index
    laps = laps[laps["Driver"].isin(eligible)]

    if laps.empty:
        return None

    # 9. Aggregate per driver
    result = laps.groupby("Driver").agg(
        corrected_median_pace=("CorrectedLapTime", "median"),
        qualifying_long_run_laps=("CorrectedLapTime", "count"),
        compound=("Compound", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN"),
    )

    # 10. Compute delta to fastest driver (feature for model)
    fastest = result["corrected_median_pace"].min()
    result["fp2_pace_delta_to_fastest"] = result["corrected_median_pace"] - fastest

    return result


# Attach for test access
extract_fp2_pace.__wrapped__ = _process_laps


def fuel_correct_lap_times(lap_times: pd.Series, stint_lap_numbers: pd.Series) -> pd.Series:
    """
    Normalise lap times to a "zero fuel burned" reference by adding back the
    time advantage gained from fuel burn-off.

    Physics:
        Each lap burned reduces the car's fuel load by ~1.7 kg, lightening the
        car and making it proportionally faster.  FUEL_CORRECTION_FACTOR (≈ 0.03 s)
        represents the lap-time benefit per lap of fuel carried.

        raw_time[k]       = true_pace  −  k × FC   (car gets faster as fuel burns)
        corrected_time[k] = raw_time[k] + k × FC   = true_pace  (fuel effect removed)

    After correction, all laps within a stint are on the same "empty-tank" basis,
    so inter-driver median comparisons reflect only raw tyre pace.
    """
    return lap_times + (stint_lap_numbers * FUEL_CORRECTION_FACTOR)

# tests/test_fp_pace.py
import pandas as pd
import numpy as np
from data.fp_pace import extract_fp2_pace, fuel_correct_lap_times


def test_fuel_correction_increases_times():
    lap_times = pd.Series([90.5, 90.3, 90.1, 89.9])
    stint_lap_numbers = pd.Series([1, 2, 3, 4])
    corrected = fuel_correct_lap_times(lap_times, stint_lap_numbers)
    assert abs(corrected.iloc[0] - (90.5 + 0.03)) < 1e-6
    assert abs(corrected.iloc[3] - (89.9 + 4 * 0.03)) < 1e-6


def test_fuel_correction_flattens_fuel_degrading_times():
    # Verify the mathematical property: correction = raw + lap_number * FC
    # If a car has no real pace change, lap_times[i] = base - lap_number * FC
    # (since the car gets faster by FC each lap as fuel burns)
    # Then: corrected[i] = (base - lap_number*FC) + lap_number*FC = base
    # Let's verify that fuel correction flattens out this fuel-burn speedup.
    from config import FUEL_CORRECTION_FACTOR
    base = 90.0
    n = 5
    # Simulating fuel burn-off (car gets FC faster each lap)
    lap_times = pd.Series([base - i * FUEL_CORRECTION_FACTOR for i in range(1, n + 1)])
    stint_lap_numbers = pd.Series(range(1, n + 1))
    corrected = fuel_correct_lap_times(lap_times, stint_lap_numbers)
    # Corrected times should all be exactly equal to 'base'
    assert all(abs(c - base) < 1e-9 for c in corrected.values)


def test_process_laps_filters_deleted_and_pit():
    # Build 9 laps: lap 0 = PitOut, lap 1 = Deleted, laps 2-8 = valid (7 clean laps)
    # LONG_RUN_MIN_LAPS = 6, so 7 valid laps should pass the threshold
    mock_laps = pd.DataFrame({
        "Driver":    ["VER"] * 9,
        "LapTime":   pd.to_timedelta([90.1, 91.0, 90.2, 90.3, 90.0, 89.9, 90.1, 90.2, 90.0], unit="s"),
        "PitOutTime": [pd.NaT] * 9,
        "PitInTime":  [pd.NaT] * 9,
        "Deleted":    [False, True, False, False, False, False, False, False, False],
        "TyreLife":   [7, 8, 9, 10, 11, 12, 13, 14, 15],
        "Compound":   ["MEDIUM"] * 9,
        "LapNumber":  [1, 2, 3, 4, 5, 6, 7, 8, 9],
    })
    # Mark lap 0 as PitOut
    mock_laps.loc[0, "PitOutTime"] = pd.Timestamp("2026-01-01 10:00:00")

    result = extract_fp2_pace.__wrapped__(mock_laps)
    assert result is not None, "Expected result but got None (check LONG_RUN_MIN_LAPS threshold)"
    assert "VER" in result.index
    # PitOut (lap 0) + Deleted (lap 1) excluded → 7 valid laps used
    assert result.loc["VER", "qualifying_long_run_laps"] == 7


def test_fp2_pace_delta_always_zero_for_fastest():
    """The fastest driver should always have delta = 0."""
    mock_laps = pd.DataFrame({
        "Driver":    ["VER"] * 8 + ["HAM"] * 8,
        "LapTime":   pd.to_timedelta([90.0] * 8 + [91.0] * 8, unit="s"),
        "PitOutTime": [pd.NaT] * 16,
        "PitInTime":  [pd.NaT] * 16,
        "Deleted":    [False] * 16,
        "TyreLife":   list(range(1, 9)) * 2,
        "Compound":   ["MEDIUM"] * 16,
        "LapNumber":  list(range(1, 9)) * 2,
    })
    result = extract_fp2_pace.__wrapped__(mock_laps)
    assert result is not None
    assert result.loc["VER", "fp2_pace_delta_to_fastest"] == 0.0
    assert result.loc["HAM", "fp2_pace_delta_to_fastest"] > 0.0

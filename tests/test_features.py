# tests/test_features.py
import pandas as pd
import numpy as np
from features.builder import impute_missing_features, validate_feature_columns
from config import FEATURE_COLUMNS


def _full_feature_df(overrides=None):
    """Build a complete 3-row feature DataFrame for testing."""
    data = {
        "fp2_pace_delta_to_fastest": [0.0, 0.5, np.nan],
        "quali_gap_to_pole":         [0.0, 0.3, 0.7],
        "grid_position":             [1.0, 3.0, np.nan],
        "team_dnf_rate_at_circuit":  [0.1, 0.0, 0.2],
        "compound_degradation_rate": [0.05, 0.05, 0.05],
        "sc_probability":            [0.3, 0.3, 0.3],
        "rain_probability":          [0.1, 0.1, 0.1],
        "circuit_overtaking_index":  [2.0, 2.0, 2.0],
        "grid_to_finish_delta_mean": [1.0, -1.0, np.nan],
        "penalty_flag":              [0.0, 0.0, 1.0],
    }
    if overrides:
        data.update(overrides)
    return pd.DataFrame(data)


def test_impute_missing_fills_with_median():
    df = _full_feature_df()
    result = impute_missing_features(df)
    # median of [0.0, 0.5] = 0.25
    assert abs(result["fp2_pace_delta_to_fastest"].iloc[2] - 0.25) < 1e-6
    # median of [1, 3] = 2.0
    assert abs(result["grid_position"].iloc[2] - 2.0) < 1e-6
    assert not result.isnull().any().any()


def test_impute_missing_falls_back_to_historical():
    # If a column is entirely NaN in df, it should fall back to the median in historical_df
    df = pd.DataFrame({
        "fp2_pace_delta_to_fastest": [np.nan, np.nan, np.nan],
        "quali_gap_to_pole":         [0.0, 0.3, 0.7],
    })
    historical_df = pd.DataFrame({
        "fp2_pace_delta_to_fastest": [1.0, 1.2, 1.4],
        "quali_gap_to_pole":         [0.0, 0.1, 0.2],
    })
    result = impute_missing_features(df, historical_df)
    # Median of [1.0, 1.2, 1.4] = 1.2
    assert all(abs(val - 1.2) < 1e-6 for val in result["fp2_pace_delta_to_fastest"])


def test_validate_feature_columns_raises_on_missing():
    df = pd.DataFrame({"fp2_pace_delta_to_fastest": [0.0]})  # missing all others
    try:
        validate_feature_columns(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing feature columns" in str(e)


def test_validate_feature_columns_passes_on_complete():
    df = _full_feature_df()
    # Should not raise
    validate_feature_columns(df)


def test_impute_does_not_change_non_nan_values():
    df = _full_feature_df()
    result = impute_missing_features(df)
    assert result["quali_gap_to_pole"].iloc[0] == 0.0
    assert result["quali_gap_to_pole"].iloc[1] == 0.3
    assert result["quali_gap_to_pole"].iloc[2] == 0.7

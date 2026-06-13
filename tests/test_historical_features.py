# tests/test_historical_features.py
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from data.historical_features import (
    _add_grid_delta, _add_dnf_rates, _add_quali_gap, enrich_training_data
)


def test_add_grid_delta():
    # Build history for VER at monaco
    # race 1: grid 1, finish 1 (delta 0)
    # race 2: grid 10, finish 5 (delta 5)
    # race 3: grid 2, finish 1 (delta 1)
    # Target race: grid 3, finish 2
    # The rolling mean of prior races should be mean([0, 5, 1]) = 2.0
    df = pd.DataFrame([
        {"season": 2021, "round": 5, "circuit_id": "monaco", "driver_code": "VER", "grid_position": 1, "finishing_position": 1},
        {"season": 2022, "round": 7, "circuit_id": "monaco", "driver_code": "VER", "grid_position": 10, "finishing_position": 5},
        {"season": 2023, "round": 6, "circuit_id": "monaco", "driver_code": "VER", "grid_position": 2, "finishing_position": 1},
        {"season": 2024, "round": 8, "circuit_id": "monaco", "driver_code": "VER", "grid_position": 3, "finishing_position": 2},
    ])
    result = _add_grid_delta(df)
    # For index 3 (season 2024), delta mean should be 2.0
    assert abs(result.loc[3, "grid_to_finish_delta_mean"] - 2.0) < 1e-6
    # For index 0, it should be 0.0 (no prior races)
    assert result.loc[0, "grid_to_finish_delta_mean"] == 0.0


def test_add_dnf_rates():
    # team_dnf_rate_at_circuit: constructor "red_bull" at "monaco"
    # season 2021: Finished
    # season 2022: Retired
    # season 2023: Finished
    # target season 2024: DNF rate should be 1 / 3 = 0.3333
    df = pd.DataFrame([
        {"season": 2021, "round": 5, "circuit_id": "monaco", "constructor_id": "red_bull", "status": "Finished"},
        {"season": 2022, "round": 7, "circuit_id": "monaco", "constructor_id": "red_bull", "status": "Retired"},
        {"season": 2023, "round": 6, "circuit_id": "monaco", "constructor_id": "red_bull", "status": "Finished"},
        {"season": 2024, "round": 8, "circuit_id": "monaco", "constructor_id": "red_bull", "status": "Finished"},
    ])
    result = _add_dnf_rates(df)
    assert abs(result.loc[3, "team_dnf_rate_at_circuit"] - (1/3)) < 1e-6


@patch("data.historical_features.get_qualifying_results")
def test_add_quali_gap(mock_get_quali):
    # Mock qualifying results for season 2024, round 8
    mock_quali = pd.DataFrame([
        {"driver_code": "VER", "quali_gap_to_pole": 0.0},
        {"driver_code": "LEC", "quali_gap_to_pole": 0.15},
    ])
    mock_get_quali.return_value = mock_quali

    df = pd.DataFrame([
        {"season": 2024, "round": 8, "driver_code": "VER"},
        {"season": 2024, "round": 8, "driver_code": "LEC"},
    ])
    result = _add_quali_gap(df)
    assert result.loc[0, "quali_gap_to_pole"] == 0.0
    assert result.loc[1, "quali_gap_to_pole"] == 0.15


@patch("data.historical_features.extract_fp2_pace")
@patch("data.historical_features.get_qualifying_results")
@patch("data.historical_features.get_compound_degradation")
@patch("data.historical_features.get_openf1_session_key")
@patch("data.historical_features.requests.get")
def test_enrich_training_data_workflow(mock_requests_get, mock_get_openf1_session, mock_get_compound_deg, mock_get_quali, mock_extract_fp2):
    # Mock all external fetches
    mock_extract_fp2.return_value = pd.DataFrame([
        {"fp2_pace_delta_to_fastest": 0.1}
    ], index=["VER"])

    mock_get_quali.return_value = pd.DataFrame([
        {"driver_code": "VER", "quali_gap_to_pole": 0.05}
    ])

    mock_get_openf1_session.return_value = [{"session_key": 9999}]
    mock_get_compound_deg.return_value = 0.04

    # Mock OpenF1 Race Control response (empty list)
    mock_response = MagicMock()
    mock_response.json.return_value = []
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    raw_df = pd.DataFrame([
        {
            "season": 2023,
            "round": 1,
            "circuit_id": "bahrain",
            "driver_code": "VER",
            "constructor_id": "red_bull",
            "grid_position": 1,
            "finishing_position": 1,
            "penalty_flag": 0,
            "status": "Finished"
        }
    ])

    enriched = enrich_training_data(raw_df, current_season=2024)
    # Check that it returns expected shape and columns without NaNs
    assert not enriched.isnull().any().any()
    assert "quali_gap_to_pole" in enriched.columns
    assert "fp2_pace_delta_to_fastest" in enriched.columns
    assert "compound_degradation_rate" in enriched.columns

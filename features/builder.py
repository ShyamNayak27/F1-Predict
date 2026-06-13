# features/builder.py
import pandas as pd
import numpy as np
from config import FEATURE_COLUMNS, FALLBACK_SC_PROBABILITY, FALLBACK_RAIN_PROBABILITY, get_overtaking_index


def build_prediction_features(
    fp2_pace_df: pd.DataFrame | None,
    qualifying_df: pd.DataFrame | None,
    reliability: dict,
    sc_probability: float,
    rain_probability: float,
    circuit_id: str,
    historical_df: pd.DataFrame,
    current_season: int,
    compound_degradation_rate: float = None,
) -> pd.DataFrame:
    """
    Assemble one row per driver with all features for live prediction.
    Falls back gracefully if any source is None.
    """
    from config import FALLBACK_COMPOUND_DEGRADATION
    if compound_degradation_rate is None:
        compound_degradation_rate = FALLBACK_COMPOUND_DEGRADATION

    # 1. Determine driver list from qualifying or FP2
    if qualifying_df is not None and not qualifying_df.empty:
        drivers = qualifying_df["driver_code"].tolist()
    elif fp2_pace_df is not None:
        drivers = fp2_pace_df.index.tolist()
    else:
        raise ValueError("Cannot build features: no qualifying or FP2 data available")

    overtaking_index = get_overtaking_index(circuit_id)

    rows = []
    for driver in drivers:
        row = {"driver_code": driver}

        # fp2_pace_delta_to_fastest
        if fp2_pace_df is not None and driver in fp2_pace_df.index:
            row["fp2_pace_delta_to_fastest"] = fp2_pace_df.loc[driver, "fp2_pace_delta_to_fastest"]
        else:
            row["fp2_pace_delta_to_fastest"] = np.nan

        # quali_gap_to_pole, grid_position, penalty_flag
        if qualifying_df is not None:
            qrow = qualifying_df[qualifying_df["driver_code"] == driver]
            if not qrow.empty:
                row["quali_gap_to_pole"] = float(qrow["quali_gap_to_pole"].iloc[0])
                row["grid_position"]     = int(qrow["grid_position"].iloc[0])
                row["penalty_flag"]      = int(qrow["penalty_flag"].iloc[0])
            else:
                row["quali_gap_to_pole"] = np.nan
                row["grid_position"]     = len(drivers) + 1
                row["penalty_flag"]      = 1
        else:
            row["quali_gap_to_pole"] = np.nan
            row["grid_position"]     = np.nan
            row["penalty_flag"]      = 0

        # team_dnf_rate_at_circuit
        constructor = None
        if qualifying_df is not None:
            qrow = qualifying_df[qualifying_df["driver_code"] == driver]
            if not qrow.empty:
                constructor = qrow["constructor_id"].iloc[0]
        if not constructor or pd.isna(constructor):
            constructor = _lookup_constructor(driver, historical_df, current_season)
            
        row["constructor_id"] = constructor
        row["team_dnf_rate_at_circuit"] = reliability.get(constructor, np.nan)

        # compound_degradation_rate
        row["compound_degradation_rate"] = compound_degradation_rate

        # shared scalars
        row["sc_probability"]          = sc_probability
        row["rain_probability"]         = rain_probability
        row["circuit_overtaking_index"] = overtaking_index

        # grid_to_finish_delta_mean from historical data
        row["grid_to_finish_delta_mean"] = _compute_grid_delta(driver, circuit_id, historical_df)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("driver_code").sort_index()
    df = impute_missing_features(df, historical_df)
    validate_feature_columns(df)
    return df


def impute_missing_features(df: pd.DataFrame, historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """Replace NaN with column median (live or historical fallback) for all numeric columns."""
    for col in df.columns:
        if df[col].dtype in [float, int, "float64", "int64"]:
            median_val = df[col].median()
            if pd.isna(median_val) and historical_df is not None and col in historical_df.columns:
                median_val = historical_df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
    return df


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def _lookup_constructor(driver_code: str, historical_df: pd.DataFrame, season: int) -> str:
    if historical_df is None or historical_df.empty:
        return "unknown"
    recent = historical_df[
        (historical_df["driver_code"] == driver_code) &
        (historical_df["season"] == season)
    ]
    if recent.empty:
        recent = historical_df[historical_df["driver_code"] == driver_code]
    if recent.empty:
        return "unknown"
    return recent.sort_values("season", ascending=False).iloc[0]["constructor_id"]


def _compute_grid_delta(driver_code: str, circuit_id: str, historical_df: pd.DataFrame) -> float:
    if historical_df is None or historical_df.empty:
        return 0.0
    rows = historical_df[
        (historical_df["driver_code"] == driver_code) &
        (historical_df["circuit_id"] == circuit_id)
    ].tail(5)
    if rows.empty:
        return 0.0
    deltas = rows["grid_position"] - rows["finishing_position"]
    return float(deltas.mean())

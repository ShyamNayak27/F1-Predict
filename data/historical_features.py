# data/historical_features.py
"""
Task 5B: Historical Feature Enrichment — Bug 6 Fix

Takes the raw historical DataFrame (from historical.py) and enriches it with
all 10 feature columns required by the model.

Sources by feature:
    - grid_position            → already in raw data
    - penalty_flag             → already in raw data
    - quali_gap_to_pole        → Jolpica qualifying endpoint (all seasons)
    - fp2_pace_delta_to_fastest→ FastF1 FP2 sessions (all seasons, cached)
    - team_dnf_rate_at_circuit → computed from raw data (rolling 3-season window)
    - compound_degradation_rate→ OpenF1 stints (2023+), default for 2021-2022
    - sc_probability           → OpenF1 race_control (2023+), default for 2021-2022
    - rain_probability         → static default (no historical weather source)
    - circuit_overtaking_index → static lookup from config
    - grid_to_finish_delta_mean→ computed from raw data (rolling prior races)
"""

import requests
import pandas as pd
import numpy as np
import time
from config import (
    JOLPICA_BASE, OPENF1_BASE, OPENF1_MIN_YEAR,
    FALLBACK_SC_PROBABILITY, FALLBACK_RAIN_PROBABILITY,
    FALLBACK_COMPOUND_DEGRADATION, JOLPICA_TO_OPENF1_CIRCUIT,
    DNF_STATUS_KEYWORDS, FEATURE_COLUMNS, TARGET_COLUMN,
    get_overtaking_index, requests_get_with_retry,
)
from data.qualifying import get_qualifying_results
from data.fp_pace import extract_fp2_pace
from data.stints import get_compound_degradation, _fetch_stints, get_openf1_session_key
from features.builder import impute_missing_features


def enrich_training_data(raw_df: pd.DataFrame, current_season: int) -> pd.DataFrame:
    """
    Master enrichment function. Adds all 10 feature columns to raw historical data.
    This is slow on first run (~30-60 min due to FastF1 FP2 fetching) but results
    are cached by FastF1 automatically.

    Args:
        raw_df: Output of build_training_dataset()
        current_season: The season we're predicting (used as upper bound)

    Returns:
        DataFrame with FEATURE_COLUMNS + TARGET_COLUMN + metadata columns
    """
    df = raw_df.copy()
    total_races = df[["season", "round"]].drop_duplicates().shape[0]
    print(f"\n[enrich] Enriching {len(df)} driver-race rows across {total_races} races...")

    # ── STEP 1: Static / computed-from-raw features ────────────────────────
    print("[enrich] Step 1/6: Circuit overtaking index...")
    df["circuit_overtaking_index"] = df["circuit_id"].apply(get_overtaking_index)

    print("[enrich] Step 2/6: Rain probability (static default — no historical source)...")
    df["rain_probability"] = FALLBACK_RAIN_PROBABILITY

    print("[enrich] Step 3/6: Grid-to-finish delta (rolling prior races per driver/circuit)...")
    df = _add_grid_delta(df)

    print("[enrich] Step 4/6: Team DNF rate at circuit (rolling 3-season window)...")
    df = _add_dnf_rates(df)

    # ── STEP 2: Jolpica qualifying (per race) ──────────────────────────────
    print("[enrich] Step 5/6: Qualifying gap to pole (Jolpica)...")
    df = _add_quali_gap(df)

    # ── STEP 3: FastF1 FP2 pace (per race — slow, cached) ─────────────────
    print("[enrich] Step 6/6: FP2 pace delta (FastF1 — this will take a while)...")
    df = _add_fp2_pace(df)

    # ── STEP 4: OpenF1 features (2023+ only) ──────────────────────────────
    print("[enrich] Step 7/8: Compound degradation rate (OpenF1 2023+)...")
    df = _add_compound_degradation(df)

    print("[enrich] Step 8/8: Safety car probability (OpenF1 2023+)...")
    df = _add_sc_probability(df)

    # ── STEP 5: Impute remaining NaN with column medians ──────────────────
    print("[enrich] Imputing remaining NaN values with column medians...")
    feature_cols_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    for col in feature_cols_present:
        if df[col].isna().any():
            median_val = df[col].median()
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"  [impute] {col}: filled {n_filled} NaN → median={median_val:.4f}")

    print(f"[enrich] [SUCCESS] Enrichment complete. Shape: {df.shape}")
    return df


# ── Per-feature helper functions ──────────────────────────────────────────────

def _add_grid_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute grid_to_finish_delta_mean: driver's mean (grid - finish) in last 5
    starts at this circuit, using only races PRIOR to the current one.
    """
    df = df.sort_values(["season", "round"]).copy()
    df["grid_to_finish_delta_mean"] = np.nan

    for idx, row in df.iterrows():
        prior = df[
            (df["driver_code"] == row["driver_code"]) &
            (df["circuit_id"] == row["circuit_id"]) &
            ((df["season"] < row["season"]) |
             ((df["season"] == row["season"]) & (df["round"] < row["round"])))
        ].tail(5)

        if prior.empty:
            df.at[idx, "grid_to_finish_delta_mean"] = 0.0
        else:
            deltas = prior["grid_position"] - prior["finishing_position"]
            df.at[idx, "grid_to_finish_delta_mean"] = float(deltas.mean())

    return df


def _add_dnf_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team_dnf_rate_at_circuit: rolling 3-season DNF rate for the
    constructor at this circuit, using only seasons PRIOR to the current one.
    """
    df = df.copy()
    df["team_dnf_rate_at_circuit"] = np.nan

    def is_dnf(status):
        return any(kw.lower() in str(status).lower() for kw in DNF_STATUS_KEYWORDS)

    df["_is_dnf"] = df["status"].apply(is_dnf)

    for idx, row in df.iterrows():
        prior = df[
            (df["constructor_id"] == row["constructor_id"]) &
            (df["circuit_id"] == row["circuit_id"]) &
            (df["season"] >= row["season"] - 3) &
            (df["season"] < row["season"])
        ]
        if prior.empty:
            # Use all circuits for this constructor if no circuit history
            prior = df[
                (df["constructor_id"] == row["constructor_id"]) &
                (df["season"] >= row["season"] - 3) &
                (df["season"] < row["season"])
            ]
        if prior.empty:
            df.at[idx, "team_dnf_rate_at_circuit"] = 0.1  # league average
        else:
            df.at[idx, "team_dnf_rate_at_circuit"] = float(prior["_is_dnf"].mean())

    df.drop(columns=["_is_dnf"], inplace=True)
    return df


def _add_quali_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch qualifying results from Jolpica for each unique (season, round) pair."""
    df = df.copy()
    df["quali_gap_to_pole"] = np.nan

    races = df[["season", "round"]].drop_duplicates().values
    total = len(races)

    for i, (season, round_num) in enumerate(races):
        print(f"  [quali] {i+1}/{total} — season {season} round {int(round_num)}...")
        quali_df = get_qualifying_results(int(season), int(round_num))
        if quali_df is None:
            continue

        for _, q in quali_df.iterrows():
            mask = (
                (df["season"] == season) &
                (df["round"] == round_num) &
                (df["driver_code"] == q["driver_code"])
            )
            df.loc[mask, "quali_gap_to_pole"] = q["quali_gap_to_pole"]

        time.sleep(0.1)  # polite rate limiting for Jolpica

    return df


def _add_fp2_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch FP2 long-run pace from FastF1 for each unique (season, round) pair.
    Slow on first run — FastF1 caches sessions to disk automatically.
    """
    df = df.copy()
    df["fp2_pace_delta_to_fastest"] = np.nan

    races = df[["season", "round"]].drop_duplicates().values
    total = len(races)

    for i, (season, round_num) in enumerate(races):
        print(f"  [fp2] {i+1}/{total} — season {season} round {int(round_num)}...")
        fp2_df = extract_fp2_pace(int(season), int(round_num))
        if fp2_df is None:
            continue  # Sprint weekend or data unavailable — NaN → imputed later

        for driver_code, fp2_row in fp2_df.iterrows():
            mask = (
                (df["season"] == season) &
                (df["round"] == round_num) &
                (df["driver_code"] == driver_code)
            )
            df.loc[mask, "fp2_pace_delta_to_fastest"] = fp2_row["fp2_pace_delta_to_fastest"]

    return df


def _add_compound_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add compound_degradation_rate from OpenF1 (2023+).
    For 2021-2022, uses FALLBACK_COMPOUND_DEGRADATION.
    One degradation value per (season, round) — shared across all drivers in that race.
    """
    df = df.copy()
    df["compound_degradation_rate"] = FALLBACK_COMPOUND_DEGRADATION

    # Only enrich 2023+ races
    openf1_races = df[df["season"] >= OPENF1_MIN_YEAR][["season", "round", "circuit_id"]].drop_duplicates()

    # Build a mapping from (season, round) → OpenF1 FP2 session_key
    session_cache = {}
    for _, row in openf1_races.iterrows():
        season    = int(row["season"])
        round_num = int(row["round"])
        circuit   = row["circuit_id"]

        openf1_name = JOLPICA_TO_OPENF1_CIRCUIT.get(circuit, circuit)
        sessions = get_openf1_session_key(
            circuit_short_name=openf1_name,
            session_name="Practice 2",
            year=season,
        )
        if sessions:
            session_cache[(season, round_num)] = sessions[0]["session_key"]

    total = len(session_cache)
    for i, ((season, round_num), sk) in enumerate(session_cache.items()):
        print(f"  [stints] {i+1}/{total} — season {season} round {round_num} (key={sk})...")
        deg = get_compound_degradation(sk)
        mask = (df["season"] == season) & (df["round"] == round_num)
        df.loc[mask, "compound_degradation_rate"] = deg

    return df


def _add_sc_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sc_probability per race from OpenF1 race_control (2023+).
    For 2021-2022, uses FALLBACK_SC_PROBABILITY.
    Uses historical SC rate for the circuit PRIOR to the current race.
    """
    df = df.copy()
    df["sc_probability"] = FALLBACK_SC_PROBABILITY

    # Build per-circuit SC event history from OpenF1 race data (2023+)
    # Map circuit → list of (session_key, total_laps, sc_events)
    circuit_sc_history = {}

    openf1_races = df[df["season"] >= OPENF1_MIN_YEAR][["season", "round", "circuit_id"]].drop_duplicates()
    openf1_races = openf1_races.sort_values(["season", "round"])

    for _, row in openf1_races.iterrows():
        season    = int(row["season"])
        round_num = int(row["round"])
        circuit   = row["circuit_id"]

        openf1_name = JOLPICA_TO_OPENF1_CIRCUIT.get(circuit, circuit)
        sessions = get_openf1_session_key(
            circuit_short_name=openf1_name,
            session_name="Race",
            year=season,
        )
        if not sessions:
            continue

        sk = sessions[0]["session_key"]

        try:
            rc_resp = requests_get_with_retry(
                f"{OPENF1_BASE}/race_control",
                params={"session_key": sk},
                timeout=15,
            )
            messages = rc_resp.json()
        except Exception as e:
            print(f"  [WARNING] [sc] Race control fetch failed for session {sk}: {e}")
            continue

        # Count SC/VSC events — check both flag field and message text (Bug 3 fix)
        sc_events = [
            m for m in messages
            if m.get("flag") in ("SAFETY CAR", "VIRTUAL SAFETY CAR")
            or "SAFETY CAR" in str(m.get("message", "")).upper()
            or "VIRTUAL SAFETY CAR" in str(m.get("message", "")).upper()
        ]

        # Fetch total race laps as denominator
        try:
            lap_resp = requests_get_with_retry(
                f"{OPENF1_BASE}/laps",
                params={"session_key": sk, "driver_number": 1},
                timeout=15,
            )
            total_laps = len(lap_resp.json())
        except Exception:
            total_laps = 57  # F1 average lap count

        if circuit not in circuit_sc_history:
            circuit_sc_history[circuit] = []
        circuit_sc_history[circuit].append({
            "season": season, "round": round_num,
            "sc_events": len(sc_events), "total_laps": max(total_laps, 1),
        })
        time.sleep(0.15)

    # Now assign SC probability to each race using ONLY prior editions at that circuit
    for idx, row in df[df["season"] >= OPENF1_MIN_YEAR].iterrows():
        circuit = row["circuit_id"]
        if circuit not in circuit_sc_history:
            continue

        history = circuit_sc_history[circuit]
        prior = [
            h for h in history
            if h["season"] < row["season"]
            or (h["season"] == row["season"] and h["round"] < row["round"])
        ]

        if not prior:
            continue  # Keep fallback

        total_sc = sum(h["sc_events"] for h in prior)
        total_laps = sum(h["total_laps"] for h in prior)
        sc_prob = min(total_sc / total_laps, 1.0) if total_laps > 0 else FALLBACK_SC_PROBABILITY
        df.at[idx, "sc_probability"] = sc_prob

    return df

# Main_Prediction_Script/pred_advanced.py
import pandas as pd
import numpy as np
import os
import sys
import joblib
import argparse
import datetime
import pytz
import requests
import warnings
import json as _json
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor
import fastf1

# Force console stdout/stderr to UTF-8 to prevent encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

# Import project config and data components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    FEATURE_COLUMNS, TARGET_COLUMN, JOLPICA_TO_OPENF1_CIRCUIT,
    OPENF1_MIN_YEAR, requests_get_with_retry, OPENF1_BASE, JOLPICA_BASE,
    FALLBACK_SC_PROBABILITY, FALLBACK_RAIN_PROBABILITY,
    FALLBACK_COMPOUND_DEGRADATION, get_overtaking_index, DNF_STATUS_KEYWORDS
)
from data.calendar import get_next_race
from data.historical import build_training_dataset
from data.qualifying import get_qualifying_results
from data.fp_pace import extract_fp2_pace, _process_laps
from data.reliability import get_team_dnf_rates
from data.safety_car import get_sc_probability
from data.weather import get_rain_probability

# Configure FastF1 cache
fastf1.Cache.enable_cache("cache/")

HISTORICAL_CACHE = "model/historical_data.joblib"
ENRICHED_CACHE = "model/enriched_data.joblib"

ROUNDS = [1, 2, 3, 4, 5, 6]
RACE_NAMES = {
    1: "Australian GP",
    2: "Chinese GP",
    3: "Japanese GP",
    4: "Miami GP",
    5: "Canadian GP",
    6: "Monaco GP",
}

CONSTRUCTOR_COLORS = {
    "alpine":    "#0093CC",
    "aston_martin": "#229971",
    "ferrari":   "#E80020",
    "haas":      "#B6BABD",
    "sauber":    "#52E252",
    "mclaren":   "#FF8000",
    "mercedes":  "#27F4D2",
    "rb":        "#6692FF",
    "red_bull":  "#3671C6",
    "williams":  "#64C4FF",
    "audi":      "#F30F30",
    "cadillac":  "#1E3B70",
    "unknown":   "#888888",
}

DRIVER_FULL_NAMES = {
    "ANT": "Kimi Antonelli",   "RUS": "George Russell",
    "HAM": "Lewis Hamilton",   "LEC": "Charles Leclerc",
    "NOR": "Lando Norris",     "PIA": "Oscar Piastri",
    "VER": "Max Verstappen",   "HAD": "Isack Hadjar",
    "LAW": "Liam Lawson",      "GAS": "Pierre Gasly",
    "OCO": "Esteban Ocon",     "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",     "SAI": "Carlos Sainz",
    "ALB": "Alexander Albon",  "COL": "Franco Colapinto",
    "HUL": "Nico Hulkenberg",  "BOT": "Valtteri Bottas",
    "BOR": "Gabriel Bortoleto","BEA": "Oliver Bearman",
    "PER": "Sergio Perez",     "LIN": "Jack Doohan",
}


def _get_constructor_color(constructor_id: str) -> str:
    cid = (constructor_id or "unknown").lower()
    for key, color in CONSTRUCTOR_COLORS.items():
        if key in cid:
            return color
    return CONSTRUCTOR_COLORS["unknown"]


class _NumpyEncoder(_json.JSONEncoder):
    """Handles numpy scalar types that the default encoder rejects."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Fetch current-season results to avoid stale team strength signal
# ---------------------------------------------------------------------------

def _fetch_current_season_results(season: int, before_round: int) -> pd.DataFrame:
    """
    Fetch completed race results from the target season for rounds < before_round.
    Used to compute up-to-date constructor points ratios for the live prediction row,
    preventing the model from relying entirely on the previous season's standings.
    """
    url = f"{JOLPICA_BASE}/{season}/results.json?limit=1000"
    try:
        resp = requests_get_with_retry(url, timeout=15)
        races = resp.json()["MRData"]["RaceTable"]["Races"]
    except Exception as e:
        print(f"  [WARNING] [standings] Could not fetch {season} in-season results: {e}")
        return pd.DataFrame()

    rows = []
    for race in races:
        if int(race["round"]) >= before_round:
            continue
        for r in race["Results"]:
            rows.append({
                "season":             season,
                "round":              int(race["round"]),
                "circuit_id":         race["Circuit"]["circuitId"],
                "constructor_id":     r["Constructor"]["constructorId"],
                "finishing_position": int(r["position"]),
                "driver_code":        r["Driver"]["code"],
            })
    if rows:
        print(f"  [standings] Fetched {len(rows)} driver-race rows from {season} in-season results "
              f"(rounds 1-{before_round - 1})")
    return pd.DataFrame(rows)


def fetch_actual_results(season: int, round_num: int) -> pd.DataFrame | None:
    """Fetch completed actual results for validation mapping."""
    url = f"{JOLPICA_BASE}/{season}/{round_num}/results.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        races = resp.json()["MRData"]["RaceTable"]["Races"]
    except Exception as e:
        print(f"  [WARNING] Could not fetch actual results for round {round_num}: {e}")
        return None
    if not races or "Results" not in races[0]:
        return None
    rows = []
    for r in races[0]["Results"]:
        rows.append({
            "actual_pos":     int(r["position"]),
            "driver_code":    r["Driver"]["code"],
            "constructor_id": r["Constructor"]["constructorId"],
            "grid":           int(r.get("grid", 0)),
            "status":         r.get("status", "Finished"),
        })
    return pd.DataFrame(rows).sort_values("actual_pos").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Check wet race - with failure counter
# ---------------------------------------------------------------------------

def check_is_wet_race(season: int, round_number: int, circuit_id: str) -> tuple[float, bool]:
    """
    Checks if a race was wet using OpenF1 race control or known lists.
    Returns (rain_probability, had_api_failure).
    rain_probability: 1.0 for wet, 0.10 for dry.
    """
    # Known wet races from 2021-2022 (pre-OpenF1)
    known_wet = {
        (2021, 2),  # Imola
        (2021, 11), # Hungary
        (2021, 12), # Spa
        (2021, 15), # Sochi
        (2021, 16), # Turkey
        (2022, 4),  # Imola
        (2022, 7),  # Monaco
        (2022, 10), # Silverstone
        (2022, 17), # Singapore
        (2022, 18), # Japan
    }
    if (season, round_number) in known_wet:
        return 1.0, False

    if season >= 2023:
        try:
            openf1_name = JOLPICA_TO_OPENF1_CIRCUIT.get(circuit_id, circuit_id)
            from data.stints import get_openf1_session_key
            sessions = get_openf1_session_key(openf1_name, "Race", season)
            if sessions:
                sk = sessions[0]["session_key"]
                resp = requests_get_with_retry(
                    f"{OPENF1_BASE}/race_control",
                    params={"session_key": sk},
                    timeout=10
                )
                messages = resp.json()
                wet_keywords = ["RAIN", "WET", "DAMP", "DRIZZLE", "INTERMEDIATE", "HEAVY WET", "WET TRACK"]
                for m in messages:
                    msg_text = str(m.get("message", "")).upper()
                    flag = str(m.get("flag", "")).upper()
                    if flag == "WET TRACK" or any(kw in msg_text for kw in wet_keywords):
                        return 1.0, False
        except Exception:
            return FALLBACK_RAIN_PROBABILITY, True

    return FALLBACK_RAIN_PROBABILITY, False


def extract_practice_pace(season: int, round_number: int) -> pd.DataFrame | None:
    """Extract practice long-run pace, falling back to FP1 if FP2 is missing (Sprint weekend)."""
    try:
        res = extract_fp2_pace(season, round_number)
        if res is not None:
            return res
    except Exception as e:
        print(f"  [practice] FP2 session load failed or data not yet available: {e}")

    print(f"  [practice] FP2 pace unavailable for {season} Round {round_number}. Trying FP1 fallback...")
    try:
        session = fastf1.get_session(season, round_number, "FP1")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = getattr(session, "laps", None)
        if laps is not None and not laps.empty:
            res = _process_laps(laps)
            if res is not None:
                print(f"  [practice] [SUCCESS] Loaded FP1 pace fallback for {len(res)} drivers.")
                return res
    except Exception as e:
        print(f"  [practice] [WARNING] FP1 fallback failed: {e}")
    return None


def _lookup_constructor_id(driver_code: str, historical_df: pd.DataFrame, season: int) -> str:
    recent = historical_df[
        (historical_df["driver_code"] == driver_code) &
        (historical_df["season"] == season)
    ]
    if recent.empty:
        recent = historical_df[historical_df["driver_code"] == driver_code]
    if recent.empty:
        return "unknown"
    return recent.sort_values("season", ascending=False).iloc[0]["constructor_id"]


def add_advanced_features(
    df: pd.DataFrame,
    historical_df: pd.DataFrame,
    current_season: int,
    current_round: int = 1,
    current_circuit_id: str = "unknown",
    in_season_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calculates constructor standings points ratios, driver track competence averages,
    and recent form metrics prior to each race weekend, preventing data leakage.
    """
    df = df.copy()

    # 1. Map positions to standard points
    all_results = historical_df.copy()
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    all_results["points_scored"] = all_results["finishing_position"].map(points_map).fillna(0)

    # Append current-season in-season results
    if in_season_df is not None and not in_season_df.empty:
        in_season_copy = in_season_df.copy()
        in_season_copy["points_scored"] = in_season_copy["finishing_position"].map(points_map).fillna(0)
        all_results = pd.concat([all_results, in_season_copy], ignore_index=True)

    # Calculate cumulative team points per season
    constructor_race_pts = all_results.groupby(["season", "round", "constructor_id"])["points_scored"].sum().reset_index()
    constructor_race_pts["cum_points"] = constructor_race_pts.groupby(["season", "constructor_id"])["points_scored"].cumsum()

    # Shift cumulative points by one round
    constructor_race_pts["prior_cum_points"] = constructor_race_pts.groupby(["season", "constructor_id"])["cum_points"].shift(1).fillna(0)

    # Round 1 fallback
    season_final_pts = constructor_race_pts.groupby(["season", "constructor_id"])["cum_points"].max().reset_index()
    prev_season_map = {}
    for idx, row in season_final_pts.iterrows():
        prev_season_map[(row["season"] + 1, row["constructor_id"])] = row["cum_points"]

    round_1_mask = constructor_race_pts["round"] == 1
    for idx, row in constructor_race_pts[round_1_mask].iterrows():
        key = (row["season"], row["constructor_id"])
        prev_pts = prev_season_map.get(key, 0)
        constructor_race_pts.at[idx, "prior_cum_points"] = prev_pts

    # Find season leader's points prior to round
    max_pts_per_race = constructor_race_pts.groupby(["season", "round"])["prior_cum_points"].transform("max")
    constructor_race_pts["team_season_points_ratio"] = (
        constructor_race_pts["prior_cum_points"] / np.where(max_pts_per_race > 0, max_pts_per_race, 1.0)
    ).clip(0.05, 1.0)

    pts_ratio_lookup = {}
    for idx, row in constructor_race_pts.iterrows():
        pts_ratio_lookup[(row["season"], row["round"], row["constructor_id"])] = row["team_season_points_ratio"]

    # 2. Driver track competence lookup
    driver_circuit_avg_lookup = {}
    all_results = all_results.sort_values(["season", "round"])
    for (driver, circuit), group in all_results.groupby(["driver_code", "circuit_id"]):
        finishes = group["finishing_position"].tolist()
        seasons = group["season"].tolist()
        rounds = group["round"].tolist()
        for i in range(len(finishes)):
            prior_finishes = finishes[:i]
            avg_finish = np.mean(prior_finishes) if prior_finishes else np.nan
            driver_circuit_avg_lookup[(seasons[i], rounds[i], driver, circuit)] = avg_finish

    # Driver overall avg lookup
    driver_overall_lookup = {}
    for driver, group in all_results.groupby("driver_code"):
        finishes = group["finishing_position"].tolist()
        seasons = group["season"].tolist()
        rounds = group["round"].tolist()
        for i in range(len(finishes)):
            prior_finishes = finishes[:i]
            driver_overall_lookup[(seasons[i], rounds[i], driver)] = np.mean(prior_finishes) if prior_finishes else np.nan

    # Recent form lookup (last 3 completed races)
    driver_recent_form_lookup = {}
    for driver, group in all_results.groupby("driver_code"):
        positions = group["finishing_position"].tolist()
        seasons = group["season"].tolist()
        rounds = group["round"].tolist()
        for i in range(len(positions)):
            window = positions[max(0, i - 3):i]
            driver_recent_form_lookup[(seasons[i], rounds[i], driver)] = (
                np.mean(window) if window else np.nan
            )

    # Map features to df rows
    team_ratios = []
    driver_circuits = []
    recent_forms = []

    for idx, row in df.iterrows():
        c_id = row.get("constructor_id")
        if pd.isna(c_id) or c_id is None:
            c_id = _lookup_constructor_id(row["driver_code"], all_results, row.get("season", current_season))

        season = row.get("season", current_season)
        round_num = row.get("round")
        if pd.isna(round_num) or round_num is None:
            round_num = current_round

        circuit_id = row.get("circuit_id")
        if pd.isna(circuit_id) or circuit_id is None:
            circuit_id = current_circuit_id

        driver = row["driver_code"]

        # Team season points ratio
        ratio = pts_ratio_lookup.get((season, round_num, c_id))
        if ratio is None:
            matching_ratios = [val for key, val in pts_ratio_lookup.items() if key[2] == c_id]
            ratio = matching_ratios[-1] if matching_ratios else 0.1
        team_ratios.append(ratio)

        # Driver circuit average finish position
        avg_finish = driver_circuit_avg_lookup.get((season, round_num, driver, circuit_id))
        if pd.isna(avg_finish):
            avg_finish = driver_overall_lookup.get((season, round_num, driver))
        if pd.isna(avg_finish):
            matching_finishes = [val for key, val in driver_circuit_avg_lookup.items() if key[2] == driver and key[3] == circuit_id and not pd.isna(val)]
            if matching_finishes:
                avg_finish = matching_finishes[-1]
            else:
                overall_finishes = [val for key, val in driver_overall_lookup.items() if key[2] == driver and not pd.isna(val)]
                avg_finish = overall_finishes[-1] if overall_finishes else 12.0
        driver_circuits.append(avg_finish)

        # Recent form
        form = driver_recent_form_lookup.get((season, round_num, driver))
        if pd.isna(form):
            teammate_forms = [
                val for (s, rnd, d), val in driver_recent_form_lookup.items()
                if s == season and rnd == round_num and d != driver and not pd.isna(val) and
                   _lookup_constructor_id(d, all_results, season) == c_id
            ]
            if teammate_forms:
                form = np.mean(teammate_forms)
            else:
                matching_finishes = all_results[all_results["constructor_id"] == c_id]["finishing_position"]
                form = matching_finishes.mean() if not matching_finishes.empty else 12.0
        recent_forms.append(form)

    df["team_season_points_ratio"] = team_ratios
    df["driver_circuit_avg_finish"] = driver_circuits
    df["driver_recent_form"] = recent_forms

    # 3. Composite score (70% team season points ratio + 30% normalized driver circuit competence)
    df["driver_track_score"] = ((22.0 - df["driver_circuit_avg_finish"]).clip(0.0, 21.0)) / 21.0
    df["car_driver_composite"] = 0.7 * df["team_season_points_ratio"] + 0.3 * df["driver_track_score"]

    # 4. Interaction feature for qualifying vs overtaking index
    df["circuit_overtaking_index"] = df["circuit_overtaking_index"].fillna(2.0)
    df["grid_x_overtaking_index"] = df["grid_position"] * df["circuit_overtaking_index"]

    return df


def compute_metrics(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> dict:
    """Merge predicted and actual finishing order; compute accuracy metrics."""
    pred = pred_df.reset_index()[["driver_code", "final_predicted_pos"]].copy()
    merged = pred.merge(actual_df[["driver_code", "actual_pos"]], on="driver_code", how="inner")
    if merged.empty:
        return {}
    merged["error"] = (merged["final_predicted_pos"] - merged["actual_pos"]).abs()
    mae              = merged["error"].mean()
    exact            = (merged["error"] == 0).sum()
    within_3         = (merged["error"] <= 3).sum()
    within_5         = (merged["error"] <= 5).sum()
    n                = len(merged)
    podium_pred      = set(merged[merged["final_predicted_pos"] <= 3]["driver_code"])
    podium_actual    = set(merged[merged["actual_pos"] <= 3]["driver_code"])
    podium_hits      = len(podium_pred & podium_actual)
    top10_pred       = set(merged[merged["final_predicted_pos"] <= 10]["driver_code"])
    top10_actual     = set(merged[merged["actual_pos"] <= 10]["driver_code"])
    top10_hits       = len(top10_pred & top10_actual)
    winner_correct   = (
        merged[merged["final_predicted_pos"] == 1]["driver_code"].values ==
        merged[merged["actual_pos"] == 1]["driver_code"].values
    ).any() if not merged.empty else False
    return {
        "mae":            round(mae, 2),
        "exact":          exact,
        "within_3":       within_3,
        "within_5":       within_5,
        "n":              n,
        "podium_hits":    podium_hits,
        "top10_hits":     top10_hits,
        "winner_correct": winner_correct,
        "merged":         merged,
    }


def make_position_weights(y: pd.Series) -> np.ndarray:
    weights = np.ones(len(y))
    y_vals = y.values if hasattr(y, "values") else np.array(y)
    weights[y_vals <= 1]  = 6.0   # Winner
    weights[y_vals <= 3]  = 4.0   # Podium
    weights[y_vals <= 10] = 2.5   # Points finish
    return weights


def run_prediction_for_round(
    round_num: int,
    enriched_df: pd.DataFrame,
    train_enriched_df: pd.DataFrame,
    model: GradientBoostingRegressor,
    model_win: GradientBoostingClassifier,
    model_dnf: GradientBoostingClassifier,
    advanced_features: list,
    current_season: int
) -> dict | None:
    print(f"\n============================================================")
    print(f"  Round {round_num}: {RACE_NAMES.get(round_num, f'Round {round_num}')}")
    print(f"============================================================")

    race_info = get_next_race(current_season, round_num)
    if race_info is None:
        print(f"  [ERROR] Race info not found for round {round_num}")
        return None
    circuit_id = race_info["circuit_id"]

    # Fetch in-season results
    print(f"  [standings] Fetching in-season results (rounds 1-{round_num-1})...")
    in_season_df = _fetch_current_season_results(current_season, round_num)

    # Practice pace
    print(f"  [practice] Loading FP2/FP1 pace...")
    practice_df = extract_practice_pace(current_season, round_num)

    # Qualifying
    print(f"  [qualifying] Loading qualifying results...")
    qualifying_df = get_qualifying_results(current_season, round_num)
    if qualifying_df is None:
        print(f"  [ERROR] No qualifying data for round {round_num}, skipping.")
        return None

    # Reliability
    reliability = get_team_dnf_rates(circuit_id, current_season)

    # SC probability
    sc_prob, _ = get_sc_probability(circuit_id)
    rain_prob  = FALLBACK_RAIN_PROBABILITY

    compound_deg = FALLBACK_COMPOUND_DEGRADATION

    # Build base feature matrix
    try:
        from features.builder import build_prediction_features
        live_features = build_prediction_features(
            fp2_pace_df=practice_df,
            qualifying_df=qualifying_df,
            reliability=reliability,
            sc_probability=sc_prob,
            rain_probability=rain_prob,
            circuit_id=circuit_id,
            historical_df=enriched_df,
            current_season=current_season,
            compound_degradation_rate=compound_deg,
        )
    except ValueError as e:
        print(f"  [ERROR] Feature build failed: {e}")
        return None

    # Add advanced features
    live_features = live_features.reset_index()
    live_features = add_advanced_features(
        live_features, enriched_df, current_season,
        current_round=round_num,
        current_circuit_id=circuit_id,
        in_season_df=in_season_df,
    )
    live_features = live_features.set_index("driver_code")

    # Predict
    X_live = live_features[advanced_features]
    preds  = model.predict(X_live)
    win_probs = model_win.predict_proba(X_live)[:, 1]
    dnf_probs = model_dnf.predict_proba(X_live)[:, 1]

    live_features["predicted_pos_raw"]   = preds
    live_features["win_probability"]     = win_probs
    live_features["dnf_risk"]            = dnf_probs

    # Blending score
    reg_score = (22.0 - preds) / 21.0
    blend_score = 0.6 * reg_score + 0.4 * win_probs
    final_blend_score = blend_score * (1.0 - 0.6 * dnf_probs)
    live_features["blend_score"] = final_blend_score

    # Assign integer rank
    live_features["final_predicted_pos"] = (
        pd.Series(final_blend_score, index=live_features.index)
        .rank(ascending=False, method="first")
        .astype(int)
    )

    # Actual results
    print(f"  [actual] Fetching actual race results...")
    actual_df = fetch_actual_results(current_season, round_num)
    if actual_df is None:
        print(f"  [WARNING] No actual results available for round {round_num}")

    metrics = compute_metrics(live_features, actual_df) if actual_df is not None else {}

    return {
        "round":        round_num,
        "race_name":    RACE_NAMES.get(round_num, f"Round {round_num}"),
        "circuit_id":   circuit_id,
        "predictions":  live_features.sort_values("final_predicted_pos"),
        "actual":       actual_df,
        "metrics":      metrics,
        "sc_prob":      sc_prob,
        "rain_prob":    rain_prob,
    }


def export_dashboard_json(results: list[dict], train_df: pd.DataFrame,
                          model, model_win, model_dnf,
                          advanced_features: list, output_path: str, season: int) -> None:
    """
    Export all prediction data to a structured JSON file for the dashboard.
    Includes per-race predictions, actuals, metrics, feature importance,
    model architecture details, and training data statistics.
    """
    # Feature importance
    feat_importance_reg = [
        {"feature": f, "importance": round(float(v), 5)}
        for f, v in sorted(
            zip(advanced_features, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
    ]
    feat_importance_win = [
        {"feature": f, "importance": round(float(v), 5)}
        for f, v in sorted(
            zip(advanced_features, model_win.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
    ]
    feat_importance_dnf = [
        {"feature": f, "importance": round(float(v), 5)}
        for f, v in sorted(
            zip(advanced_features, model_dnf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
    ]

    # Training data stats
    train_stats = {
        "total_rows": int(len(train_df)),
        "seasons": sorted([int(s) for s in train_df["season"].unique()]),
        "num_seasons": int(train_df["season"].nunique()),
        "num_circuits": int(train_df["circuit_id"].nunique()) if "circuit_id" in train_df.columns else 0,
        "num_drivers": int(train_df["driver_code"].nunique()) if "driver_code" in train_df.columns else 0,
    }

    # Per-race data
    races = []
    all_winner_correct = []
    for r in results:
        m = r["metrics"]
        predictions_df = r["predictions"]
        actual_df = r["actual"]

        # Build driver rows
        drivers = []
        if m and "merged" in m:
            merged = m["merged"]
            for _, row in predictions_df.iterrows():
                drv = row.name if hasattr(row, 'name') else str(row.get("driver_code", ""))
                mrow = merged[merged["driver_code"] == drv]
                actual_pos = int(mrow["actual_pos"].values[0]) if not mrow.empty else None
                error = int(mrow["error"].values[0]) if not mrow.empty else None

                c_id = str(row.get("constructor_id", "unknown"))
                drivers.append({
                    "code":           drv,
                    "full_name":      DRIVER_FULL_NAMES.get(drv, drv),
                    "constructor_id": c_id,
                    "team_color":     _get_constructor_color(c_id),
                    "predicted_pos":  int(row["final_predicted_pos"]),
                    "actual_pos":     actual_pos,
                    "error":          error,
                    "win_prob":       round(float(row.get("win_probability", 0)), 4),
                    "dnf_risk":       round(float(row.get("dnf_risk", 0)), 4),
                    "blend_score":    round(float(row.get("blend_score", 0)), 4),
                    "grid_position":  int(row.get("grid_position", 0)) if not pd.isna(row.get("grid_position", 0)) else 0,
                    "team_pts_ratio": round(float(row.get("team_season_points_ratio", 0)), 4),
                    "recent_form":    round(float(row.get("driver_recent_form", 12)), 2) if not pd.isna(row.get("driver_recent_form", float("nan"))) else None,
                    "circuit_avg":    round(float(row.get("driver_circuit_avg_finish", 12)), 2) if not pd.isna(row.get("driver_circuit_avg_finish", float("nan"))) else None,
                    "car_composite":  round(float(row.get("car_driver_composite", 0)), 4),
                })
        else:
            for drv, row in predictions_df.iterrows():
                c_id = str(row.get("constructor_id", "unknown"))
                drivers.append({
                    "code":           drv,
                    "full_name":      DRIVER_FULL_NAMES.get(drv, drv),
                    "constructor_id": c_id,
                    "team_color":     _get_constructor_color(c_id),
                    "predicted_pos":  int(row["final_predicted_pos"]),
                    "actual_pos":     None,
                    "error":          None,
                    "win_prob":       round(float(row.get("win_probability", 0)), 4),
                    "dnf_risk":       round(float(row.get("dnf_risk", 0)), 4),
                    "blend_score":    round(float(row.get("blend_score", 0)), 4),
                    "grid_position":  int(row.get("grid_position", 0)) if not pd.isna(row.get("grid_position", 0)) else 0,
                    "team_pts_ratio": round(float(row.get("team_season_points_ratio", 0)), 4),
                    "recent_form":    round(float(row.get("driver_recent_form", 12)), 2) if not pd.isna(row.get("driver_recent_form", float("nan"))) else None,
                    "circuit_avg":    round(float(row.get("driver_circuit_avg_finish", 12)), 2) if not pd.isna(row.get("driver_circuit_avg_finish", float("nan"))) else None,
                    "car_composite":  round(float(row.get("car_driver_composite", 0)), 4),
                })

        has_actuals = m and "merged" in m and actual_df is not None
        winner_correct = bool(m.get("winner_correct", False)) if m else False
        all_winner_correct.append(winner_correct)

        races.append({
            "round":       r["round"],
            "race_name":   r["race_name"],
            "circuit_id":  r["circuit_id"],
            "flag":        "", # Removed emoji flags
            "sc_prob":     round(r["sc_prob"], 3),
            "rain_prob":   round(r["rain_prob"], 2),
            "has_actuals": has_actuals,
            "metrics": {
                "mae":          m.get("mae") if m else None,
                "exact":        int(m["exact"]) if m else None,
                "within_3":     int(m["within_3"]) if m else None,
                "within_5":     int(m["within_5"]) if m else None,
                "n":            int(m["n"]) if m else None,
                "podium_hits":  int(m["podium_hits"]) if m else None,
                "top10_hits":   int(m["top10_hits"]) if m else None,
                "winner_correct": winner_correct,
            } if m else None,
            "drivers": drivers,
        })

    # Season-level aggregate
    valid = [r for r in results if r["metrics"]]
    aggregate = {
        "total_races":       len(results),
        "races_with_actuals": len(valid),
        "winners_correct":   sum(1 for r in valid if r["metrics"].get("winner_correct")),
        "podium_hits":       sum(r["metrics"]["podium_hits"] for r in valid),
        "podium_total":      len(valid) * 3,
        "top10_hits":        sum(r["metrics"]["top10_hits"] for r in valid),
        "top10_total":       len(valid) * 10,
        "avg_mae":           round(float(np.mean([r["metrics"]["mae"] for r in valid])), 2) if valid else None,
        "exact_total":       sum(r["metrics"]["exact"] for r in valid),
        "within3_total":     sum(r["metrics"]["within_3"] for r in valid),
        "n_total":           sum(r["metrics"]["n"] for r in valid),
        "winner_streak":     all_winner_correct,
    }

    # Model architecture description
    model_info = {
        "name": "Blended Multi-Task GBR + GBC Ensemble",
        "description": (
            "Three scikit-learn gradient boosting models trained on 5 seasons (2021-2025) "
            "of Formula 1 race data. A position regressor, a binary winner classifier, and a "
            "DNF risk classifier are combined into a single blended ranking score."
        ),
        "blend_formula": "blend_score = (0.6 x reg_score + 0.4 x win_prob) * (1 - 0.6 * dnf_risk)",
        "components": [
            {
                "name": "Position Regressor (GBR)",
                "type": "GradientBoostingRegressor",
                "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
                "purpose": "Predicts raw finishing position (1-22)",
                "training": "Weighted: winner 6x, podium 4x, top-10 2.5x to prioritize front-runners",
                "weight_in_blend": 0.6,
            },
            {
                "name": "Winner Classifier (GBC)",
                "type": "GradientBoostingClassifier",
                "params": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
                "purpose": "Predicts probability of winning the race (binary)",
                "training": "Class-weighted 5:1 (winner vs non-winner) to correct class imbalance",
                "weight_in_blend": 0.4,
            },
            {
                "name": "DNF Risk Classifier (GBC)",
                "type": "GradientBoostingClassifier",
                "params": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
                "purpose": "Predicts probability of driver retirement/DNF",
                "training": "Trained on DNF status labels from standard reliability keywords",
                "weight_in_blend": -0.6,
            },
        ],
        "features": advanced_features,
        "feature_descriptions": {
            "fp2_pace_delta_to_fastest": "FP2 median long-run lap time gap to fastest driver (fuel-corrected, seconds)",
            "quali_gap_to_pole": "Qualifying time gap to pole position (seconds; 0 for pole sitter)",
            "grid_position": "Race starting position (1-22; pit lane starts = 21)",
            "team_dnf_rate_at_circuit": "Constructor DNF rate at this circuit over last 3 seasons",
            "compound_degradation_rate": "Seconds of pace loss per lap on primary tire compound",
            "sc_probability": "Historical safety car probability at this circuit",
            "rain_probability": "Rain probability at race time",
            "circuit_overtaking_index": "Overtaking difficulty: 1=easy (Monza/Baku), 2=medium, 3=hard (Monaco/Hungary)",
            "grid_to_finish_delta_mean": "Driver's mean position gained/lost from grid in last 5 race starts",
            "penalty_flag": "1 if driver starts from pit lane or has a grid penalty, else 0",
            "team_season_points_ratio": "Constructor points ratio vs. leader (includes in-season results to avoid data leakage)",
            "driver_circuit_avg_finish": "Driver's historical average finishing position at this circuit",
            "car_driver_composite": "0.7 x team_points_ratio + 0.3 x driver_track_score (composite pace signal)",
            "driver_recent_form": "Rolling average finishing position over last 3 completed races",
            "grid_x_overtaking_index": "Interaction term: grid position x circuit overtaking difficulty",
        },
        "cross_validation": "GroupKFold(n_splits=10) grouped by (season, round) - prevents same-race leakage",
        "training_data": train_stats,
        "feature_importance": {
            "regressor": feat_importance_reg,
            "winner_classifier": feat_importance_win,
            "dnf_classifier": feat_importance_dnf,
        },
        "data_sources": [
            {"name": "Jolpica (Ergast mirror)", "url": "https://api.jolpi.ca/ergast/f1", "usage": "Historical race results, qualifying, constructors standings"},
            {"name": "OpenF1 API", "url": "https://api.openf1.org/v1", "usage": "FP2 lap times, safety car events, race control messages"},
            {"name": "FastF1", "url": "https://github.com/theOehrly/Fast-F1", "usage": "Practice session telemetry and lap data"},
            {"name": "OpenWeatherMap", "url": "https://openweathermap.org/api", "usage": "Race-day rain probability forecast"},
        ],
        "tech_stack": [
            "Python 3.11",
            "scikit-learn 1.4 (GradientBoostingRegressor, GradientBoostingClassifier, GroupKFold)",
            "XGBoost (comparison baseline)",
            "pandas 2.x, numpy 1.26",
            "FastF1 3.x",
            "requests (with exponential-backoff retry)",
            "joblib (model/data caching)",
        ],
        "known_limitations": [
            "DNFs caused by racing incidents (crashes, mechanical failures) are inherently unpredictable.",
            "Round 1 predictions are weakest: no in-season data exists, so constructor strength relies entirely on prior-year standings.",
            "Monaco is a structural edge case: extreme circuit difficulty means qualifying position dominates.",
            "Rain probability uses a fallback of 10% when OpenWeatherMap data is unavailable post-race.",
        ],
    }

    output = {
        "generated_at": datetime.datetime.now().isoformat(),
        "season": season,
        "aggregate": aggregate,
        "races": races,
        "model": model_info,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(output, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"\n[SUCCESS] Dashboard JSON written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="F1 Race Prediction Script")
    parser.add_argument("--season",  type=int, default=2026, help="Target season year")
    parser.add_argument("--round",   type=int, default=2,    help="Target round number")
    parser.add_argument("--retrain", action="store_true",    help="Rebuild and enrich historical data cache")
    parser.add_argument("--compare", action="store_true",    help="Run base vs advanced vs XGBoost model CV comparison (slower)")
    parser.add_argument("--dashboard", action="store_true",  help="Run full season simulation and export to dashboard JSON")
    args = parser.parse_args()

    current_season = args.season
    current_round  = args.round

    print("\n" + "=" * 70)
    print(f"       F1 PREDICTION MODEL - SEASON {current_season}")
    print("=" * 70)

    # 1. Load Enriched training dataset
    if not os.path.exists(ENRICHED_CACHE) or args.retrain:
        print("\nEnriching historical dataset (this can take a few minutes)...")
        if not os.path.exists(HISTORICAL_CACHE):
            historical_df = build_training_dataset(current_season)
            joblib.dump(historical_df, HISTORICAL_CACHE)
        else:
            historical_df = joblib.load(HISTORICAL_CACHE)

        from data.historical_features import enrich_training_data
        enriched_df = enrich_training_data(historical_df, current_season)
        joblib.dump(enriched_df, ENRICHED_CACHE)
    else:
        print("\nLoading cached enriched training dataset...")
        enriched_df = joblib.load(ENRICHED_CACHE)

    if len(enriched_df) < 200:
        print("Training data cache is small. Rebuilding full database...")
        historical_df = build_training_dataset(current_season)
        joblib.dump(historical_df, HISTORICAL_CACHE)
        from data.historical_features import enrich_training_data
        enriched_df = enrich_training_data(historical_df, current_season)
        joblib.dump(enriched_df, ENRICHED_CACHE)

    # Apply historical wet-weather classifications if retraining
    if args.retrain:
        print("Re-applying historical wet-weather classifications (--retrain mode)...")
        unique_races = enriched_df[["season", "round", "circuit_id"]].drop_duplicates()
        wet_failures = 0
        for _, r in unique_races.iterrows():
            is_wet, had_failure = check_is_wet_race(int(r["season"]), int(r["round"]), r["circuit_id"])
            wet_failures += had_failure
            mask = (enriched_df["season"] == r["season"]) & (enriched_df["round"] == r["round"])
            enriched_df.loc[mask, "rain_probability"] = is_wet
        if wet_failures:
            print(f"  [WARNING] {wet_failures}/{len(unique_races)} historical wet-race lookups failed (defaulted to dry)")
    else:
        print("Using cached wet-weather classifications (use --retrain to refresh)")

    # 2. Add advanced features to training data
    print("Computing advanced features on training data...")
    train_enriched_df = add_advanced_features(
        enriched_df, enriched_df, current_season,
        current_round=1,
        current_circuit_id="unknown",
        in_season_df=None,
    )

    # Advanced feature list
    base_features = FEATURE_COLUMNS
    advanced_features = base_features + [
        "team_season_points_ratio",
        "driver_circuit_avg_finish",
        "car_driver_composite",
        "driver_recent_form",
        "grid_x_overtaking_index"
    ]

    train_clean = train_enriched_df.dropna(subset=[TARGET_COLUMN, "grid_position"]).copy()
    X_base = train_clean[base_features]
    X_adv  = train_clean[advanced_features]
    y      = train_clean[TARGET_COLUMN]

    # --- Mode: Dashboard Export ---
    if args.dashboard:
        print("\nTraining final advanced models on full historical dataset for dashboard...")
        model_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        sample_weights = make_position_weights(y)
        model_reg.fit(X_adv, y, sample_weight=sample_weights)

        y_win = (y == 1).astype(int)
        model_win = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        model_win.fit(X_adv, y_win, sample_weight=(y_win * 5 + 1))

        y_dnf = train_clean["status"].apply(lambda s: any(kw.lower() in str(s).lower() for kw in DNF_STATUS_KEYWORDS)).astype(int)
        model_dnf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        model_dnf.fit(X_adv, y_dnf)

        results = []
        for rnd in ROUNDS:
            res = run_prediction_for_round(
                rnd, enriched_df, train_enriched_df,
                model_reg, model_win, model_dnf,
                advanced_features, current_season
            )
            if res is not None:
                results.append(res)

        dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
        json_path = os.path.join(dashboard_dir, "race_results.json")
        export_dashboard_json(
            results=results,
            train_df=train_enriched_df,
            model=model_reg,
            model_win=model_win,
            model_dnf=model_dnf,
            advanced_features=advanced_features,
            output_path=json_path,
            season=current_season
        )
        
        # Copy race_results.json to root directory
        root_dir = os.path.join(os.path.dirname(__file__), "..")
        landing_json_path = os.path.join(root_dir, "race_results.json")
        try:
            import shutil
            shutil.copy2(json_path, landing_json_path)
            print(f"[SUCCESS] Copied race_results.json to root: {landing_json_path}")
        except Exception as e:
            print(f"[WARNING] Failed to copy race_results.json to root: {e}")
        return

    # --- Mode: Cross-Validation Comparison ---
    if args.compare:
        print("\n" + "-" * 68)
        print("  MODEL CROSS-VALIDATION COMPARISON (GroupKFold, k=10)")
        print("  Groups by race (season, round) - no same-race leakage")
        print("-" * 68)

        groups = train_clean["season"].astype(str) + "_" + train_clean["round"].astype(str)
        gkf = GroupKFold(n_splits=10)

        def _eval_model(model_type, X, groups):
            errs = []
            win_hits = 0
            top10_hits = 0
            podium_hits = 0
            total_win = 0
            total_top10 = 0
            total_podium = 0

            for train_idx, test_idx in gkf.split(X, y, groups):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                statuses_tr = train_clean.iloc[train_idx]["status"]

                if model_type == "xgb":
                    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0)
                    model.fit(X_tr, y_tr)
                    scores = -model.predict(X_te)
                else:  # base GBR
                    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
                    model.fit(X_tr, y_tr)
                    scores = -model.predict(X_te)

                test_df = pd.DataFrame({
                    "score": scores,
                    "actual": y_te.values,
                    "group": groups.iloc[test_idx].values
                })

                for race, rdf in test_df.groupby("group"):
                    ranked_pred = rdf["score"].rank(ascending=False, method="first").astype(int)
                    rdf = rdf.copy()
                    rdf["ranked_pred"] = ranked_pred
                    
                    errs.extend(np.abs(rdf["ranked_pred"] - rdf["actual"]))

                    actual_winner = rdf[rdf["actual"] == 1]
                    pred_winner = rdf[rdf["ranked_pred"] == 1]
                    if not actual_winner.empty and not pred_winner.empty:
                        win_hits += int(actual_winner.index[0] == pred_winner.index[0])
                        total_win += 1

                    pod_pred = set(rdf[rdf["ranked_pred"] <= 3].index)
                    pod_act = set(rdf[rdf["actual"] <= 3].index)
                    podium_hits += len(pod_pred & pod_act)
                    total_podium += 3

                    t10_pred = set(rdf[rdf["ranked_pred"] <= 10].index)
                    t10_act = set(rdf[rdf["actual"] <= 10].index)
                    top10_hits += len(t10_pred & t10_act)
                    total_top10 += 10

            mae = np.mean(errs)
            winner_acc = win_hits / total_win if total_win > 0 else 0
            podium_pct = podium_hits / total_podium if total_podium > 0 else 0
            top10_pct = top10_hits / total_top10 if total_top10 > 0 else 0

            return mae, winner_acc, podium_pct, top10_pct

        mae_base, win_base, pod_base, t10_base = _eval_model("base", X_base, groups)
        mae_xgb, win_xgb, pod_xgb, t10_xgb = _eval_model("xgb", X_adv, groups)

        print(f"  Base GBR:  MAE={mae_base:.2f} | Win={win_base*100:.1f}% | Pod={pod_base*100:.1f}% | Top10={t10_base*100:.1f}%")
        print(f"  Adv XGB:   MAE={mae_xgb:.2f} | Win={win_xgb*100:.1f}% | Pod={pod_xgb*100:.1f}% | Top10={t10_xgb*100:.1f}%")
        print("-" * 68)

    # --- Mode: Single GP Prediction ---
    race_info = get_next_race(current_season, current_round)
    if race_info is None:
        print(f"Round {current_round} not found in the {current_season} calendar.")
        return
    current_circuit_id = race_info["circuit_id"]
    print(f"Target GP: {race_info['race_name']} ({current_circuit_id})")

    # Fetch current-season standings results
    print(f"\nFetching {current_season} in-season standings (rounds before Round {current_round})...")
    in_season_df = _fetch_current_season_results(current_season, current_round)

    # Calculate advanced features on training
    train_clean_features = add_advanced_features(
        enriched_df, enriched_df, current_season,
        current_round=current_round,
        current_circuit_id=current_circuit_id,
        in_season_df=None,
    )

    # Extract Live Features for Target GP
    print(f"\nFetching live practice pace for Round {current_round}...")
    practice_pace_df = extract_practice_pace(current_season, current_round)

    print(f"Fetching qualifying grid for Round {current_round}...")
    qualifying_df = get_qualifying_results(current_season, current_round)

    if qualifying_df is None:
        print("Qualifying results not available yet. Cannot generate predictions.")
        return

    print(f"Fetching team reliability/DNF statistics...")
    reliability = get_team_dnf_rates(current_circuit_id, current_season)

    print(f"Fetching safety car probability for {current_circuit_id}...")
    sc_prob, has_sc = get_sc_probability(current_circuit_id)
    if not has_sc:
        print(f"  [WARNING] No OpenF1 SC data - using fallback ({FALLBACK_SC_PROBABILITY})")

    print(f"Fetching rain forecast for {race_info['race_name']}...")
    rain_prob, has_rain = get_rain_probability(
        race_info["lat"], race_info["lon"], race_info["race_datetime_utc"]
    )
    if not has_rain:
        print(f"  [WARNING] No weather data - using fallback ({FALLBACK_RAIN_PROBABILITY})")

    compound_deg = FALLBACK_COMPOUND_DEGRADATION

    # Build prediction row features
    from features.builder import build_prediction_features
    live_features = build_prediction_features(
        fp2_pace_df=practice_pace_df,
        qualifying_df=qualifying_df,
        reliability=reliability,
        sc_probability=sc_prob,
        rain_probability=rain_prob,
        circuit_id=current_circuit_id,
        historical_df=train_clean_features,
        current_season=current_season,
        compound_degradation_rate=compound_deg,
    )

    # Add advanced features to live dataframe
    live_features = live_features.reset_index()
    live_features = add_advanced_features(
        live_features, train_clean_features, current_season,
        current_round=current_round,
        current_circuit_id=current_circuit_id,
        in_season_df=in_season_df,
    )
    live_features = live_features.set_index("driver_code")

    # Train advanced models
    print("\nTraining final advanced regressor & classifiers...")
    model_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    sample_weights = make_position_weights(y)
    model_reg.fit(X_adv, y, sample_weight=sample_weights)

    y_win = (y == 1).astype(int)
    model_win = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model_win.fit(X_adv, y_win, sample_weight=(y_win * 5 + 1))

    y_dnf = train_clean["status"].apply(lambda s: any(kw.lower() in str(s).lower() for kw in DNF_STATUS_KEYWORDS)).astype(int)
    model_dnf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model_dnf.fit(X_adv, y_dnf)

    # Predict
    X_live = live_features[advanced_features]
    preds = model_reg.predict(X_live)
    win_probs = model_win.predict_proba(X_live)[:, 1]
    dnf_probs = model_dnf.predict_proba(X_live)[:, 1]

    live_features["predicted_pos_raw"] = preds
    live_features["win_probability"] = win_probs
    live_features["dnf_risk"] = dnf_probs

    # Blending
    reg_score = (22.0 - preds) / 21.0
    blend_score = 0.6 * reg_score + 0.4 * win_probs
    final_blend_score = blend_score * (1.0 - 0.6 * dnf_probs)
    live_features["blend_score"] = final_blend_score

    # Assign integer rank
    live_features["final_predicted_pos"] = (
        pd.Series(final_blend_score, index=live_features.index)
        .rank(ascending=False, method="first")
        .astype(int)
    )

    final_order = live_features.sort_values("final_predicted_pos")

    # Print Prediction Table
    print("\n" + "=" * 80)
    print(f"   PREDICTED FINISHING ORDER - {race_info['race_name'].upper()}")
    print("   Blended Multi-Task Model (GBR + Win Prob + DNF Risk)")
    print(f"   SC prob: {sc_prob:.2f}  |  Rain prob: {rain_prob:.2f}")
    print("=" * 80)
    print(f"  {'P':<4}{'DRV':<6}{'Grid':<6}{'Raw GBR':<10}{'Win %':<10}{'DNF %':<10}{'Blend Score'}")
    print("  " + "-" * 70)
    for idx, row in final_order.iterrows():
        win_str = f"{row['win_probability']*100:.1f}%"
        dnf_str = f"{row['dnf_risk']*100:.1f}%"
        print(
            f"  {int(row['final_predicted_pos']):<4}"
            f"{idx:<6}"
            f"Grid {int(row['grid_position']):<2}  "
            f"{row['predicted_pos_raw']:<10.1f}"
            f"{win_str:<10}"
            f"{dnf_str:<10}"
            f"{row['blend_score']:.3f}"
        )
    print("=" * 80)

    # Feature Importance report
    print("\nAdvanced Feature Importances (Regressor):")
    importances = sorted(zip(advanced_features, model_reg.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {feat:<26} {imp:.4f}  {bar}")

    if not args.compare:
        print("\n  (Tip: run with --compare to see a GroupKFold CV model comparison)")


if __name__ == "__main__":
    main()

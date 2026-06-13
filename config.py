# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
JOLPICA_BASE    = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE     = "https://api.openf1.org/v1"
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"

# --- Model ---
TRAINING_SEASONS       = 5           # How many past seasons to train on
FUEL_CORRECTION_FACTOR = 0.03        # Seconds per lap for fuel load correction
LONG_RUN_MIN_LAPS      = 6           # Minimum stint length to qualify as a long run
DNF_STATUS_KEYWORDS    = [
    "Retired", "Engine", "Gearbox", "Accident", "Collision",
    "Hydraulics", "Electrical", "Mechanical", "Suspension",
    "Brakes", "Power Unit", "Transmission", "Overheating",
]

# --- Fallback ---
FALLBACK_SC_PROBABILITY          = 0.30   # 30% SC probability when no history available
FALLBACK_RAIN_PROBABILITY        = 0.10   # 10% rain probability when weather fetch fails
FALLBACK_COMPOUND_DEGRADATION    = 0.05   # seconds/lap tire degradation when no OpenF1 data

# --- Feature columns (must match exactly between training and prediction) ---
FEATURE_COLUMNS = [
    "fp2_pace_delta_to_fastest",   # driver FP2 median long-run vs fastest driver, fuel-corrected (s)
    "quali_gap_to_pole",           # qualifying time gap to pole (seconds); 0 for pole sitter
    "grid_position",               # integer 1..20 (pit lane starts = 21)
    "team_dnf_rate_at_circuit",    # DNF rate for constructor at this circuit, last 3 seasons
    "compound_degradation_rate",   # seconds of pace loss per lap on primary compound
    "sc_probability",              # historical SC probability at this circuit
    "rain_probability",            # from OpenWeatherMap at race time (default if unavailable)
    "circuit_overtaking_index",    # 1=easy (Monza), 2=medium, 3=hard (Monaco)
    "grid_to_finish_delta_mean",   # driver's mean position gained/lost from grid in last 5 starts
    "penalty_flag",                # 1 if driver starts from pit lane or has grid penalty, else 0
]

TARGET_COLUMN = "finishing_position"

# --- Circuit ID mapping: Jolpica circuitId → OpenF1 circuit_short_name ---
# Built from cross-referencing both live APIs (2023-2025 data)
JOLPICA_TO_OPENF1_CIRCUIT = {
    "albert_park": "Melbourne",
    "shanghai":    "Shanghai",
    "suzuka":      "Suzuka",
    "bahrain":     "Sakhir",
    "jeddah":      "Jeddah",
    "miami":       "Miami",
    "imola":       "Imola",
    "monaco":      "Monte Carlo",
    "catalunya":   "Catalunya",
    "villeneuve":  "Montreal",
    "red_bull_ring": "Spielberg",
    "silverstone": "Silverstone",
    "spa":         "Spa-Francorchamps",
    "hungaroring": "Hungaroring",
    "zandvoort":   "Zandvoort",
    "monza":       "Monza",
    "baku":        "Baku",
    "marina_bay":  "Singapore",
    "americas":    "Austin",
    "rodriguez":   "Mexico City",
    "interlagos":  "Interlagos",
    "vegas":       "Las Vegas",
    "losail":      "Lusail",
    "yas_marina":  "Yas Marina Circuit",
}

# --- Circuit overtaking index (derived from historical pass data) ---
# 1=easy overtaking (DRS highway), 2=medium, 3=hard (street/twisty)
HARD_OVERTAKING_CIRCUITS = {
    "monaco", "hungaroring", "zandvoort", "marina_bay",  # Bug 4 fixed: "marina_bay" not "singapore"
}
EASY_OVERTAKING_CIRCUITS = {
    "monza", "baku", "spa", "jeddah", "bahrain", "albert_park",
}

def get_overtaking_index(circuit_id: str) -> int:
    if circuit_id in HARD_OVERTAKING_CIRCUITS:
        return 3
    if circuit_id in EASY_OVERTAKING_CIRCUITS:
        return 1
    return 2

# --- OpenF1 data coverage: only available from 2023 onwards ---
OPENF1_MIN_YEAR = 2023


def requests_get_with_retry(url: str, params: dict = None, timeout: int = 10, max_retries: int = 4, backoff_factor: float = 2.0):
    import time
    import requests
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    sleep_time = float(retry_after) if retry_after else (backoff_factor ** attempt + 1.0)
                except ValueError:
                    sleep_time = backoff_factor ** attempt + 1.0
                print(f"[WARNING] [API 429] Rate limit hit. Sleeping {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(sleep_time)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            sleep_time = backoff_factor ** attempt + 1.0
            time.sleep(sleep_time)

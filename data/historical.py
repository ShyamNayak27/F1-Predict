# data/historical.py
import requests
import pandas as pd
from config import JOLPICA_BASE, TRAINING_SEASONS, DNF_STATUS_KEYWORDS


def build_training_dataset(current_season: int) -> pd.DataFrame:
    """
    Fetch race results from Jolpica for TRAINING_SEASONS prior seasons.
    Returns a flat DataFrame where each row is one driver's result at one race.
    Raw data only — call enrich_training_data() from historical_features.py next.
    """
    all_rows = []
    for season in range(current_season - TRAINING_SEASONS, current_season):
        print(f"  [historical] Fetching season {season}...")
        url = f"{JOLPICA_BASE}/{season}/results.json?limit=1000"
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            races = resp.json()["MRData"]["RaceTable"]["Races"]
        except Exception as e:
            print(f"  [WARNING] [historical] Failed to fetch {season}: {e}")
            continue

        for race in races:
            rows = parse_race_results(
                race["Results"],
                circuit_id=race["Circuit"]["circuitId"],
                season=season,
                round_num=int(race["round"]),
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"  [historical] Loaded {len(df)} driver-race rows across {df['season'].nunique()} seasons.")
    return df


def parse_race_results(results: list, circuit_id: str, season: int, round_num: int) -> list:
    """Parse one race's results list from Jolpica into flat row dicts."""
    position_map = encode_dnf_position(results)
    num_starters = len(results)

    rows = []
    for r in results:
        code = r["Driver"]["code"]
        grid = int(r.get("grid", 0))
        if grid == 0:
            # Pit lane start — remap to beyond last grid position
            grid = num_starters + 1
            penalty = 1
        else:
            penalty = 0

        rows.append({
            "season":              season,
            "round":               round_num,
            "circuit_id":          circuit_id,
            "driver_code":         code,
            "constructor_id":      r["Constructor"]["constructorId"],
            "grid_position":       grid,
            "finishing_position":  position_map[code],
            "penalty_flag":        penalty,
            "laps_completed":      int(r.get("laps", 0)),
            "status":              r.get("status", ""),
        })
    return rows


def encode_dnf_position(results: list) -> dict:
    """
    Finishers get their stated position. DNFs are ranked after finishers,
    ordered by how many laps they completed (more laps = better DNF rank).
    Returns: dict of driver_code → final finishing_position integer
    """
    def is_dnf(r):
        return any(kw.lower() in r.get("status", "").lower() for kw in DNF_STATUS_KEYWORDS)

    finishers = [r for r in results if not is_dnf(r)]
    dnfs      = [r for r in results if is_dnf(r)]

    # Sort DNFs: most laps completed = classified first among DNFs
    dnfs_sorted = sorted(dnfs, key=lambda r: int(r.get("laps", 0)), reverse=True)

    position_map = {}
    for r in finishers:
        position_map[r["Driver"]["code"]] = int(r["position"])

    offset = len(finishers) + 1
    for i, r in enumerate(dnfs_sorted):
        position_map[r["Driver"]["code"]] = offset + i

    return position_map

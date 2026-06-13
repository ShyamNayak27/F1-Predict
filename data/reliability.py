# data/reliability.py
import requests
from config import JOLPICA_BASE, DNF_STATUS_KEYWORDS
from collections import defaultdict


def get_team_dnf_rates(circuit_id: str, current_season: int, num_seasons: int = 3) -> dict[str, float]:
    """
    Return a dict of {constructor_id: dnf_rate} for the given circuit
    over the past num_seasons seasons.
    dnf_rate = DNFs / race starts for that constructor at this circuit.
    """
    season_start = current_season - num_seasons
    all_results = []

    for season in range(season_start, current_season):
        url = f"{JOLPICA_BASE}/{season}/circuits/{circuit_id}/results.json?limit=200"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            races = resp.json()["MRData"]["RaceTable"]["Races"]
        except Exception as e:
            print(f"[WARNING] [reliability] Failed to fetch {season} at {circuit_id}: {e}")
            continue

        for race in races:
            for r in race.get("Results", []):
                all_results.append({
                    "constructor": r["Constructor"]["constructorId"],
                    "is_dnf": any(
                        kw.lower() in r.get("status", "").lower()
                        for kw in DNF_STATUS_KEYWORDS
                    ),
                })

    if not all_results:
        return {}

    starts = defaultdict(int)
    dnfs   = defaultdict(int)
    for r in all_results:
        c = r["constructor"]
        starts[c] += 1
        if r["is_dnf"]:
            dnfs[c] += 1

    return {c: dnfs[c] / starts[c] for c in starts if starts[c] > 0}

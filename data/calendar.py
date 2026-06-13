# data/calendar.py
import requests
import datetime
import pytz
from config import JOLPICA_BASE


def get_next_race(season: int = None, round_number: int = None) -> dict | None:
    """
    Fetch the Jolpica race calendar and return structured info for the next upcoming race
    (or the specified round_number).
    Returns None if all races in the season are in the past (and no round_number specified)
    or the API fails.
    """
    if season is None:
        season = datetime.datetime.now(pytz.utc).year

    url = f"{JOLPICA_BASE}/{season}.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARNING] [calendar] Failed to fetch season calendar: {e}")
        return None

    races = resp.json()["MRData"]["RaceTable"]["Races"]
    now_utc = datetime.datetime.now(pytz.utc)

    for race in races:
        current_round = int(race["round"])
        if round_number is not None and current_round != round_number:
            continue

        race_dt = _parse_jolpica_datetime(race["date"], race["time"])
        if round_number is not None or race_dt >= now_utc:
            fp1   = race.get("FirstPractice", {})
            fp2   = race.get("SecondPractice", {})
            quali = race.get("Qualifying", {})
            # Sprint weekends have no SecondPractice — detect and flag
            is_sprint = "Sprint" in race and "SecondPractice" not in race
            return {
                "round":              current_round,
                "race_name":          race["raceName"],
                "circuit_id":         race["Circuit"]["circuitId"],
                "country":            race["Circuit"]["Location"]["country"],
                "lat":                float(race["Circuit"]["Location"]["lat"]),
                "lon":                float(race["Circuit"]["Location"]["long"]),
                "race_datetime_utc":  race_dt,
                "fp1_datetime_utc":   _parse_jolpica_datetime(fp1.get("date", ""), fp1.get("time", "")) if fp1 else None,
                "fp2_datetime_utc":   _parse_jolpica_datetime(fp2.get("date", ""), fp2.get("time", "")) if fp2 else None,
                "quali_datetime_utc": _parse_jolpica_datetime(quali.get("date", ""), quali.get("time", "")) if quali else None,
                "is_sprint_weekend":  is_sprint,
            }
    return None  # All races in the past or specified round not found


def _parse_jolpica_datetime(date_str: str, time_str: str) -> datetime.datetime | None:
    """Parse Jolpica date+time strings into a UTC-aware datetime."""
    if not date_str:
        return None
    time_str = time_str.rstrip("Z") if time_str else "00:00:00"
    dt = datetime.datetime.fromisoformat(f"{date_str}T{time_str}")
    return dt.replace(tzinfo=pytz.utc)

# data/weather.py
import requests
import datetime
import pytz
from config import OPENWEATHER_API_KEY, OPENWEATHER_BASE, FALLBACK_RAIN_PROBABILITY


def get_rain_probability(lat: float, lon: float, race_datetime_utc: datetime.datetime) -> tuple[float, bool]:
    """
    Returns (rain_probability, is_live_data).
    rain_probability is between 0.0 and 1.0.
    is_live_data is False if we fell back to the default.
    """
    if not OPENWEATHER_API_KEY:
        print("[WARNING] [weather] No API key — using fallback rain probability")
        return FALLBACK_RAIN_PROBABILITY, False

    url = f"{OPENWEATHER_BASE}/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        forecasts = resp.json().get("list", [])
    except Exception as e:
        print(f"[WARNING] [weather] Forecast fetch failed: {e} — using fallback")
        return FALLBACK_RAIN_PROBABILITY, False

    if not forecasts:
        print("[WARNING] [weather] Empty forecast — using fallback")
        return FALLBACK_RAIN_PROBABILITY, False

    # Find the 3-hour window closest to race start time
    best = None
    best_delta = float("inf")
    for fc in forecasts:
        fc_dt = datetime.datetime.fromtimestamp(fc["dt"], tz=pytz.utc)
        delta = abs((fc_dt - race_datetime_utc).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best = fc

    if best is None:
        print("[WARNING] [weather] No forecast window found — using fallback")
        return FALLBACK_RAIN_PROBABILITY, False

    rain_prob = best.get("pop", 0.0)   # pop = probability of precipitation, 0-1
    print(f"  [weather] Rain probability at race time: {rain_prob:.0%} (closest forecast ±{best_delta/3600:.1f}h)")
    return float(rain_prob), True

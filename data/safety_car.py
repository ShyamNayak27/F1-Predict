# data/safety_car.py
from config import OPENF1_BASE, FALLBACK_SC_PROBABILITY, JOLPICA_TO_OPENF1_CIRCUIT, OPENF1_MIN_YEAR, requests_get_with_retry


def get_sc_probability(circuit_id: str, num_seasons: int = 3) -> tuple[float, bool]:
    """
    Query OpenF1 race_control for historical SC/VSC events at this circuit.
    Returns (sc_probability, is_live_data).
    sc_probability = SC events / total race laps across last num_seasons editions.

    Bug 1 fix: Converts Jolpica circuit_id to OpenF1 circuit_short_name via mapping.
    Bug 3 fix: Checks both flag field AND message text for SC events.
    """
    # Bug 1 fix: map Jolpica circuit_id → OpenF1 circuit_short_name
    openf1_name = JOLPICA_TO_OPENF1_CIRCUIT.get(circuit_id, circuit_id)

    # Find race sessions at this circuit (most recent num_seasons)
    url = f"{OPENF1_BASE}/sessions"
    params = {"circuit_short_name": openf1_name, "session_name": "Race"}
    try:
        resp = requests_get_with_retry(url, params=params, timeout=10)
        sessions = resp.json()
    except Exception as e:
        print(f"[WARNING] [safety_car] Session lookup failed for {circuit_id} ({openf1_name}): {e} — using fallback")
        return FALLBACK_SC_PROBABILITY, False

    # OpenF1 only has 2023+ data
    sessions = [s for s in sessions if s.get("year", 0) >= OPENF1_MIN_YEAR]
    sessions = sorted(sessions, key=lambda s: s.get("date_start", ""), reverse=True)[:num_seasons]

    if not sessions:
        print(f"[WARNING] [safety_car] No OpenF1 sessions found for {circuit_id} — using fallback")
        return FALLBACK_SC_PROBABILITY, False

    total_sc_events = 0
    total_laps = 0

    for s in sessions:
        sk = s["session_key"]
        try:
            rc_resp = requests_get_with_retry(
                f"{OPENF1_BASE}/race_control",
                params={"session_key": sk},
                timeout=15,
            )
            messages = rc_resp.json()
        except Exception:
            continue

        # Bug 3 fix: check both flag field AND message text
        sc_events = [
            m for m in messages
            if m.get("flag") in ("SAFETY CAR", "VIRTUAL SAFETY CAR")
            or "SAFETY CAR" in str(m.get("message", "")).upper()
            or "VIRTUAL SAFETY CAR" in str(m.get("message", "")).upper()
        ]
        total_sc_events += len(sc_events)

        # Fetch total laps for denominator
        try:
            lap_resp = requests_get_with_retry(
                f"{OPENF1_BASE}/laps",
                params={"session_key": sk, "driver_number": 1},
                timeout=15,
            )
            total_laps += len(lap_resp.json())
        except Exception:
            total_laps += 57   # F1 average lap count as fallback

    if total_laps == 0:
        return FALLBACK_SC_PROBABILITY, False

    sc_prob = total_sc_events / total_laps
    print(f"  [safety_car] {circuit_id}: {total_sc_events} SC events over {total_laps} laps → p={sc_prob:.3f}")
    return min(sc_prob, 1.0), True

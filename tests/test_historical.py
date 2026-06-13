# tests/test_historical.py
from data.historical import parse_race_results, encode_dnf_position


def test_encode_dnf_position_ordered_by_last_lap():
    """DNFs ranked after finishers, ordered by how many laps they completed."""
    finishers = [
        {"Driver": {"code": "VER"}, "position": "1", "status": "Finished", "laps": "57"},
        {"Driver": {"code": "LEC"}, "position": "2", "status": "Finished", "laps": "57"},
    ]
    dnfs = [
        {"Driver": {"code": "HAM"}, "position": "17", "status": "Engine",  "laps": "42"},
        {"Driver": {"code": "ALO"}, "position": "18", "status": "Retired", "laps": "30"},
    ]
    result = encode_dnf_position(finishers + dnfs)
    assert result["VER"] == 1
    assert result["LEC"] == 2
    assert result["HAM"] == 3  # more laps → classified first among DNFs
    assert result["ALO"] == 4


def test_pit_lane_start_remapped():
    """grid_position = 0 in Jolpica = pit lane start → remap to num_starters + 1."""
    raw = [
        {"Driver": {"code": "VER"}, "Constructor": {"constructorId": "red_bull"},
         "grid": "1", "position": "1", "status": "Finished", "laps": "57"},
        {"Driver": {"code": "RUS"}, "Constructor": {"constructorId": "mercedes"},
         "grid": "0", "position": "15", "status": "Finished", "laps": "57"},
    ]
    rows = parse_race_results(raw, circuit_id="monaco", season=2024, round_num=8)
    rus_row = next(r for r in rows if r["driver_code"] == "RUS")
    assert rus_row["grid_position"] == len(raw) + 1
    assert rus_row["penalty_flag"] == 1


def test_finished_drivers_not_marked_dnf():
    """Lapped drivers with status 'Lapped' should not be treated as DNF."""
    raw = [
        {"Driver": {"code": "VER"}, "Constructor": {"constructorId": "red_bull"},
         "grid": "1", "position": "1", "status": "Finished", "laps": "57"},
        {"Driver": {"code": "ZHO"}, "Constructor": {"constructorId": "sauber"},
         "grid": "17", "position": "11", "status": "Lapped", "laps": "56"},
    ]
    rows = parse_race_results(raw, circuit_id="bahrain", season=2024, round_num=1)
    zho_row = next(r for r in rows if r["driver_code"] == "ZHO")
    # Lapped is NOT in DNF_STATUS_KEYWORDS → should be classified at stated position
    assert zho_row["finishing_position"] == 11

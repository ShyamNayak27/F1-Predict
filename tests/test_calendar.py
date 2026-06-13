# tests/test_calendar.py
from unittest.mock import patch, MagicMock
from data.calendar import get_next_race
import datetime

MOCK_CALENDAR = {
    "MRData": {
        "RaceTable": {
            "Races": [
                {
                    "round": "10",
                    "raceName": "Monaco Grand Prix",
                    "date": "2099-05-25",
                    "time": "13:00:00Z",
                    "Circuit": {
                        "circuitId": "monaco",
                        "Location": {"lat": "43.7347", "long": "7.4205", "country": "Monaco"},
                    },
                    "FirstPractice":  {"date": "2099-05-22", "time": "11:30:00Z"},
                    "SecondPractice": {"date": "2099-05-22", "time": "15:00:00Z"},
                    "Qualifying":     {"date": "2099-05-24", "time": "14:00:00Z"},
                }
            ]
        }
    }
}

MOCK_SPRINT_CALENDAR = {
    "MRData": {
        "RaceTable": {
            "Races": [
                {
                    "round": "2",
                    "raceName": "Chinese Grand Prix",
                    "date": "2099-03-23",
                    "time": "07:00:00Z",
                    "Circuit": {
                        "circuitId": "shanghai",
                        "Location": {"lat": "31.3389", "long": "121.22", "country": "China"},
                    },
                    "FirstPractice":    {"date": "2099-03-21", "time": "03:30:00Z"},
                    "Qualifying":       {"date": "2099-03-22", "time": "07:00:00Z"},
                    "Sprint":           {"date": "2099-03-22", "time": "03:00:00Z"},
                    "SprintQualifying": {"date": "2099-03-21", "time": "07:30:00Z"},
                }
            ]
        }
    }
}


def test_get_next_race_returns_correct_fields():
    with patch("data.calendar.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = MOCK_CALENDAR
        race = get_next_race(season=2099)
    assert race["circuit_id"] == "monaco"
    assert race["round"] == 10
    assert race["lat"] == 43.7347
    assert race["lon"] == 7.4205
    assert isinstance(race["race_datetime_utc"], datetime.datetime)
    assert isinstance(race["fp2_datetime_utc"], datetime.datetime)
    assert race["is_sprint_weekend"] is False


def test_get_next_race_detects_sprint_weekend():
    with patch("data.calendar.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = MOCK_SPRINT_CALENDAR
        race = get_next_race(season=2099)
    assert race["circuit_id"] == "shanghai"
    assert race["fp2_datetime_utc"] is None
    assert race["is_sprint_weekend"] is True


def test_get_next_race_returns_none_if_all_past():
    past_calendar = {
        "MRData": {"RaceTable": {"Races": [
            {
                "round": "1", "raceName": "Past Race", "date": "2000-01-01", "time": "13:00:00Z",
                "Circuit": {"circuitId": "past", "Location": {"lat": "0", "long": "0", "country": "X"}},
                "FirstPractice":  {"date": "2000-01-01", "time": "10:00:00Z"},
                "SecondPractice": {"date": "2000-01-01", "time": "14:00:00Z"},
                "Qualifying":     {"date": "2000-01-01", "time": "12:00:00Z"},
            }
        ]}}
    }
    with patch("data.calendar.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = past_calendar
        race = get_next_race(season=2000)
    assert race is None

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import personas
from personas import format_profile, generate_best_persona, RankingResult, ScenarioResponse
from profiles import Archetype, Attributes, MobilityLevel, Profile, WorkArrangement


def make_profile(home_type: str | None = None, home_ownership: str | None = None, workplace_ev_charging: bool | None = None, telecommute_days: str | None = None, commute_freq: str | None = None) -> Profile:
    return Profile(
        archetype = Archetype.FLEXIBLE_COMMUTER,
        attributes = Attributes(
            is_caregiver = False,
            mobility_level = MobilityLevel.MODERATE,
            work_arrangement = WorkArrangement.HYBRID,
            schedule_irregular = False
        ),
        age_group = "35-44",
        household_income = "$75,000-$99,999",
        employment_status = "Employed full-time",
        student_status = None,
        workplace_ev_charging = workplace_ev_charging,
        workplace_loc = None,
        telecommute_days = telecommute_days,
        commute_freq = commute_freq,
        school_freq = None,
        school_type = None,
        home_type = home_type,
        home_ownership = home_ownership,
        household_size = 3,
        gender = None,
        race_ethnicity = None,
        trips = []
    )


def test_format_profile_includes_charging_access():
    profile = make_profile(home_type = "Single-family house", home_ownership = "Own", workplace_ev_charging = True, telecommute_days = "2", commute_freq = "5 days a week")
    text = format_profile(profile)
    assert "Home Type: Single-family house" in text
    assert "Home Ownership: Own" in text
    assert "Workplace EV Charging: available" in text
    assert "Telecommute Days: 2" in text
    assert "Commute Frequency: 5 days a week" in text


def test_format_profile_workplace_charging_not_available():
    text = format_profile(make_profile(workplace_ev_charging = False))
    assert "Workplace EV Charging: not available" in text


def test_format_profile_omits_unset_charging_access():
    text = format_profile(make_profile())
    assert "Home Type:" not in text
    assert "Workplace EV Charging:" not in text


def test_generate_best_persona_picks_tournament_winner(monkeypatch):
    ids = iter(["persona0", "persona1", "persona2"])
    monkeypatch.setattr(personas, "SCENARIOS", {Archetype.FLEXIBLE_COMMUTER: ["s1", "s2"]})
    monkeypatch.setattr(personas, "DISPOSITIONS", {Archetype.FLEXIBLE_COMMUTER: ["d0", "d1", "d2"]})
    monkeypatch.setattr(personas, "generate_persona", lambda profile, client, disposition: next(ids))
    monkeypatch.setattr(personas, "answer_scenario", lambda persona, scenario, client: ScenarioResponse(action = "A", reasoning = "r"))
    monkeypatch.setattr(personas, "rank_personas", lambda profile, scenario, responses, client: RankingResult(reasoning = "r", ranking = [1, 2, 3]))

    artifact = generate_best_persona(make_profile(), client = None)
    assert artifact.best_index == 0
    assert artifact.personas[0].persona == "persona0"
    assert artifact.personas[0].disposition == "d0"
    assert artifact.personas[0].score == 4
    assert len(artifact.personas) == 3
    assert len(artifact.rankings) == 2
    assert artifact.rankings[0].ranking == [0, 1, 2]

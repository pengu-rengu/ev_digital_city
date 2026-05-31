import pandas as pd
from typing import NamedTuple
from enum import Enum
from pydantic import BaseModel
from labels import *
import json
from pathlib import Path

class Archetype(Enum):
    STUDENT = "Student"
    WORKING_ADULT = "Working Adult"
    FLEXIBLE_ADULT = "Flexible Adult"

class MobilityLevel(Enum):
    LOW = "Low Mobility"
    MODERATE = "Moderate Mobility"
    HIGH = "High Mobility"

class WorkArrangement(Enum):
    IN_PERSON = "In-Person"
    HYBRID = "Hybrid"
    REMOTE = "Remote"

class Attributes(BaseModel):
    is_caregiver: bool
    mobility_level: MobilityLevel
    work_arrangement: WorkArrangement | None
    schedule_irregular: bool

class Trip(BaseModel):
    origin_activity: str | None
    dest_activity: str | None
    travel_time: float
    distance: float
    departure_time: str
    arrival_time: str

class Profile(BaseModel):
    archetype: Archetype
    attributes: Attributes
    age_group: str
    household_income: str
    employment_status: str | None
    telecommute_days: str | None
    commute_freq: str | None
    school_freq: str | None
    school_type: str | None
    trips: list[Trip]

def ev_rows(person_df: pd.DataFrame, household_df: pd.DataFrame, vehicle_df: pd.DataFrame) -> pd.DataFrame:
    ev_household_ids = vehicle_df[vehicle_df["FUELTYPE"] == 5]["HOUSEHOLD_ID"].unique()

    ev_household_df = household_df[household_df["HOUSEHOLD_ID"].isin(ev_household_ids)]
    ev_person_df = person_df[person_df["HOUSEHOLD_ID"].isin(ev_household_ids)]
    
    return pd.merge(ev_household_df, ev_person_df, on = "HOUSEHOLD_ID")

def trips_for(person_id: int, trip_df: pd.DataFrame) -> list[Trip]:
    person_trips = trip_df[trip_df["PERSON_ID"] == person_id]
    trips = []
    for row in person_trips.itertuples(index = False):
        trip = Trip(
            origin_activity = ACTIVITY_LABELS[row.O_ACTIVITY],
            dest_activity = ACTIVITY_LABELS[row.D_ACTIVITY],
            travel_time = row.REPORTED_TRAVEL_TIME,
            distance = row.DISTANCE,
            departure_time = row.DEPARTURE_TIME_HHMM,
            arrival_time = row.ARRIVAL_TIME_HHMM,
        )
        trips.append(trip)
    return trips

def classify_archetype(row: NamedTuple) -> Archetype:
    employment_status = row.EMPLOYMENT_STATUS
    if employment_status == 0:
        return Archetype.WORKING_ADULT
    elif employment_status == 6 or row.STUDENT_STATUS == 1:
        return Archetype.STUDENT
    
    return Archetype.FLEXIBLE_ADULT



def irregular_schedule_person_ids(rows: pd.DataFrame, trip_df: pd.DataFrame) -> set[int]:
    def hhmm_to_minutes(s: str) -> int:
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    ev_trips = trip_df[trip_df["PERSON_ID"].isin(rows["PERSON_ID"])]
    work_trips = ev_trips[(ev_trips["O_ACTIVITY"] == 2) | (ev_trips["D_ACTIVITY"] == 2)].copy()
    work_trips["dep_min"] = work_trips["DEPARTURE_TIME_HHMM"].map(hhmm_to_minutes)
    work_trips["arr_min"] = work_trips["ARRIVAL_TIME_HHMM"].map(hhmm_to_minutes)

    dep_mean, dep_std = work_trips["dep_min"].mean(), work_trips["dep_min"].std()
    arr_mean, arr_std = work_trips["arr_min"].mean(), work_trips["arr_min"].std()

    dep_z = (work_trips["dep_min"] - dep_mean) / dep_std
    arr_z = (work_trips["arr_min"] - arr_mean) / arr_std

    flagged = work_trips[(dep_z.abs() > 2.0) | (arr_z.abs() > 2.0)]
    return set(flagged["PERSON_ID"])

def top_bottom_miles(rows: pd.DataFrame, trip_df: pd.DataFrame) -> tuple[set[int], set[int]]:
    person_ids = rows["PERSON_ID"]
    person_trips = trip_df[trip_df["PERSON_ID"].isin(person_ids)]
    miles_per_person = person_trips.groupby("PERSON_ID", as_index = False)["DISTANCE"].sum()

    top_cutoff = miles_per_person["DISTANCE"].quantile(0.75)
    bottom_cutoff = miles_per_person["DISTANCE"].quantile(0.25)

    top_ids = set(miles_per_person[miles_per_person["DISTANCE"] >= top_cutoff]["PERSON_ID"])
    bottom_ids = set(miles_per_person[miles_per_person["DISTANCE"] <= bottom_cutoff]["PERSON_ID"])

    return top_ids, bottom_ids

def build_attributes(row: NamedTuple, archetype: Archetype, caregiving_household_ids: set[int], top_miles_ids: set[int], bottom_miles_ids: set[int], irregular_person_ids: set[int],) -> Attributes:
    is_caregiver = row.HOUSEHOLD_ID in caregiving_household_ids
    schedule_irregular = row.PERSON_ID in irregular_person_ids

    if row.PERSON_ID in bottom_miles_ids:
        mobility_level = MobilityLevel.LOW
    elif row.PERSON_ID in top_miles_ids:
        mobility_level = MobilityLevel.HIGH
    else:
        mobility_level = MobilityLevel.MODERATE

    work_arrangement: WorkArrangement | None = None
    if archetype == Archetype.WORKING_ADULT:
        days = row.J1_TELECOMMUTE_DAYS
        if days == -9 or days == 0:
            work_arrangement = WorkArrangement.IN_PERSON
        elif days == 5:
            work_arrangement = WorkArrangement.REMOTE
        else:
            work_arrangement = WorkArrangement.HYBRID

    return Attributes(
        is_caregiver = is_caregiver,
        mobility_level = mobility_level,
        work_arrangement = work_arrangement,
        schedule_irregular = schedule_irregular,
    )

def build_profiles(rows: pd.DataFrame, trip_df: pd.DataFrame) -> list[Profile]:
    caregiving_household_ids = set(rows[rows["AGE"] < 12]["HOUSEHOLD_ID"])
    top_miles_ids, bottom_miles_ids = top_bottom_miles(rows, trip_df)
    irregular_person_ids = irregular_schedule_person_ids(rows, trip_df)

    profiles = []
    for row in rows.itertuples(index = False):
        if row.LICENSE != 1:
            continue

        trips = trips_for(row.PERSON_ID, trip_df)
        if not trips:
            continue
        
        archetype = classify_archetype(row)
        attributes = build_attributes(row, archetype, caregiving_household_ids, top_miles_ids, bottom_miles_ids, irregular_person_ids)

        profile = Profile(
            archetype = archetype,
            attributes = attributes,
            age_group = AGE_GROUP_LABELS[row.AGE_GROUP],
            household_income = INCOME_LABELS[row.HH_INCOME_DETAILED],
            employment_status = EMPLOYMENT_LABELS[row.EMPLOYMENT_STATUS],
            telecommute_days = TELECOMMUTE_DAYS_LABELS[row.J1_TELECOMMUTE_DAYS],
            commute_freq = COMMUTE_FREQ_LABELS[row.J1_COMMUTE_FREQ],
            school_freq = SCHOOL_FREQ_LABELS[row.SCHOOL_FREQ],
            school_type = SCHOOL_TYPE_LABELS[row.SCHOOL_TYPE],
            trips = trips
        )
        profiles.append(profile)

    return profiles


if __name__ == "__main__":

    person_df = pd.read_csv("data/person.csv")
    household_df = pd.read_csv("data/household.csv")
    vehicle_df = pd.read_csv("data/vehicle.csv")
    trip_df = pd.read_csv("data/trip.csv")
    rows = ev_rows(person_df, household_df, vehicle_df)

    profiles = build_profiles(rows, trip_df)
    profiles_json = [json.loads(profile.model_dump_json()) for profile in profiles]

    Path("artifacts").mkdir(exist_ok = True)
    with open("artifacts/profiles.json", "w") as file:
        json.dump(profiles_json, file, indent = 2)

import pandas as pd
from collections import Counter
from typing import NamedTuple
from enum import Enum
from pydantic import BaseModel
from labels import *
import json
from pathlib import Path

class Archetype(Enum):
    NON_COMMUTER = "Non-Commuter"
    PARENT_COMMUTER = "Parent Commuter"
    FLEXIBLE_COMMUTER = "Flexible Commuter"
    RIGID_COMMUTER = "Rigid Commuter"

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

class Vehicle(BaseModel):
    fuel_type: str | None
    body_type: str | None

class Trip(BaseModel):
    origin_activity: str | None
    dest_activity: str | None
    travel_time: float
    distance: float
    departure_time: str
    arrival_time: str
    vehicle: Vehicle | None
    travelers_total: int
    travel_mode: str | None
    origin_coords: tuple[float, float] | None
    dest_coords: tuple[float, float] | None

class Profile(BaseModel):
    archetype: Archetype
    attributes: Attributes
    age_group: str
    household_income: str
    employment_status: str | None
    student_status: str | None
    workplace_ev_charging: bool | None
    workplace_loc: str | None
    telecommute_days: str | None
    commute_freq: str | None
    school_freq: str | None
    school_type: str | None
    home_type: str | None
    home_ownership: str | None
    household_size: int
    gender: str | None
    race_ethnicity: str | None
    trips: list[Trip]

def vehicle_lookup(vehicle_df: pd.DataFrame) -> dict[tuple[int, int], Vehicle]:
    lookup: dict[tuple[int, int], Vehicle] = {}
    for row in vehicle_df.itertuples(index = False):
        lookup[(row.HOUSEHOLD_ID, row.VEHNUM)] = Vehicle(
            fuel_type = FUELTYPE_LABELS.get(row.FUELTYPE),
            body_type = BODYTYPE_LABELS.get(row.BODYTYPE),
        )
    return lookup

def lookup_coords(tpb_taz: int, bmc_taz: int, tpb_centroids: dict[int, tuple[float, float]], bmc_centroids: dict[int, tuple[float, float]]) -> tuple[float, float] | None:
    if tpb_taz != -9 and tpb_taz in tpb_centroids:
        return tpb_centroids[tpb_taz]
    if bmc_taz != -9 and bmc_taz in bmc_centroids:
        return bmc_centroids[bmc_taz]
    return None

def trips_for(person_id: int, household_id: int, trip_df: pd.DataFrame, vehicles: dict[tuple[int, int], Vehicle], tpb_centroids: dict[int, tuple[float, float]], bmc_centroids: dict[int, tuple[float, float]]) -> list[Trip]:
    person_trips = trip_df[trip_df["PERSON_ID"] == person_id]
    trips = []
    for row in person_trips.itertuples(index = False):
        vehicle = vehicles.get((household_id, row.MODE_HH_VEHICLE))
        origin_coords = lookup_coords(row.O_TPB_TAZ, row.O_BMC_TAZ, tpb_centroids, bmc_centroids)
        dest_coords = lookup_coords(row.D_TPB_TAZ, row.D_BMC_TAZ, tpb_centroids, bmc_centroids)
        trip = Trip(
            origin_activity = ACTIVITY_LABELS[row.O_ACTIVITY],
            dest_activity = ACTIVITY_LABELS[row.D_ACTIVITY],
            travel_time = row.REPORTED_TRAVEL_TIME,
            distance = row.DISTANCE,
            departure_time = row.DEPARTURE_TIME_HHMM,
            arrival_time = row.ARRIVAL_TIME_HHMM,
            vehicle = vehicle,
            travelers_total = row.TRAVELERS_TOTAL,
            travel_mode = TRAVEL_MODE_LABELS[row.TRAVEL_MODE],
            origin_coords = origin_coords,
            dest_coords = dest_coords,
        )
        trips.append(trip)
    return trips

def classify_archetype(row: NamedTuple, trips: list[Trip]) -> Archetype:
    if row.EMPLOYMENT_STATUS != 0 or row.J1_WORKPLACE_LOC in (3, -9):
        return Archetype.NON_COMMUTER
    if row.HHSIZE >= 4:
        return Archetype.PARENT_COMMUTER
    if sum(1 for trip in trips if trip.dest_activity not in ("Home", "Work")) >= 2:
        return Archetype.FLEXIBLE_COMMUTER
    return Archetype.RIGID_COMMUTER

def irregular_schedule_person_ids(rows: pd.DataFrame, trip_df: pd.DataFrame) -> set[int]:
    def hhmm_to_mins(s: str) -> int:
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    person_trips = trip_df[trip_df["PERSON_ID"].isin(rows["PERSON_ID"])]
    work_trips = person_trips[(person_trips["O_ACTIVITY"] == 2) | (person_trips["D_ACTIVITY"] == 2)].copy()
    work_trips["dep_time"] = work_trips["DEPARTURE_TIME_HHMM"].map(hhmm_to_mins)
    work_trips["arr_time"] = work_trips["ARRIVAL_TIME_HHMM"].map(hhmm_to_mins)

    departure_mean = work_trips["dep_time"].mean()
    departure_std = work_trips["dep_time"].std()

    arrival_mean = work_trips["arr_time"].mean()
    arrival_std = work_trips["arr_time"].std()

    dep_z_score = (work_trips["dep_time"] - departure_mean) / departure_std
    arr_z_score = (work_trips["arr_time"] - arrival_mean) / arrival_std

    return set(work_trips[(dep_z_score.abs() > 2.0) | (arr_z_score.abs() > 2.0)]["PERSON_ID"])

def top_bottom_miles(rows: pd.DataFrame, trip_df: pd.DataFrame) -> tuple[set[int], set[int]]:
    person_trips = trip_df[trip_df["PERSON_ID"].isin(rows["PERSON_ID"])]
    miles_per_person = person_trips.groupby("PERSON_ID", as_index = False)["DISTANCE"].sum()

    top_cutoff = miles_per_person["DISTANCE"].quantile(0.75)
    bottom_cutoff = miles_per_person["DISTANCE"].quantile(0.25)

    top_ids = set(miles_per_person[miles_per_person["DISTANCE"] >= top_cutoff]["PERSON_ID"])
    bottom_ids = set(miles_per_person[miles_per_person["DISTANCE"] <= bottom_cutoff]["PERSON_ID"])

    return top_ids, bottom_ids

def build_attributes(row: NamedTuple, caregiving_household_ids: set[int], top_miles_ids: set[int], bottom_miles_ids: set[int], irregular_person_ids: set[int],) -> Attributes:
    is_caregiver = row.HOUSEHOLD_ID in caregiving_household_ids
    schedule_irregular = row.PERSON_ID in irregular_person_ids

    if row.PERSON_ID in bottom_miles_ids:
        mobility_level = MobilityLevel.LOW
    elif row.PERSON_ID in top_miles_ids:
        mobility_level = MobilityLevel.HIGH
    else:
        mobility_level = MobilityLevel.MODERATE

    work_arrangement: WorkArrangement | None = None
    if row.EMPLOYMENT_STATUS == 0:
        days = row.J1_TELECOMMUTE_DAYS
        if days == 5 or row.J1_WORKPLACE_LOC == 3:
            work_arrangement = WorkArrangement.REMOTE
        elif days == -9 or days == 0:
            work_arrangement = WorkArrangement.IN_PERSON
        else:
            work_arrangement = WorkArrangement.HYBRID

    return Attributes(
        is_caregiver = is_caregiver,
        mobility_level = mobility_level,
        work_arrangement = work_arrangement,
        schedule_irregular = schedule_irregular,
    )

def build_profiles(df: pd.DataFrame, trip_df: pd.DataFrame, vehicle_df: pd.DataFrame, tpb_centroids: dict[int, tuple[float, float]], bmc_centroids: dict[int, tuple[float, float]]) -> list[Profile]:
    vehicles = vehicle_lookup(vehicle_df)
    caregiving_household_ids = set(df[(df["AGE"] < 18) & (df["LICENSE"] != 1)]["HOUSEHOLD_ID"])
    top_miles_ids, bottom_miles_ids = top_bottom_miles(df, trip_df)
    irregular_person_ids = irregular_schedule_person_ids(df, trip_df)

    profiles = []
    for row in df.itertuples(index = False):
        if row.LICENSE != 1:
            continue

        trips = trips_for(row.PERSON_ID, row.HOUSEHOLD_ID, trip_df, vehicles, tpb_centroids, bmc_centroids)
        if not trips:
            continue
        
        archetype = classify_archetype(row, trips)
        attributes = build_attributes(row, caregiving_household_ids, top_miles_ids, bottom_miles_ids, irregular_person_ids)

        profile = Profile(
            archetype = archetype,
            attributes = attributes,
            age_group = AGE_GROUP_LABELS[row.AGE_GROUP],
            household_income = INCOME_LABELS[row.HH_INCOME_DETAILED],
            employment_status = EMPLOYMENT_LABELS[row.EMPLOYMENT_STATUS],
            student_status = STUDENT_STATUS_LABELS[row.STUDENT_STATUS],
            workplace_ev_charging = {0: False, 1: True}.get(row.J1_BENEFITS_EV_CHARGING),
            workplace_loc = WORKPLACE_LOC_LABELS[row.J1_WORKPLACE_LOC],
            telecommute_days = TELECOMMUTE_DAYS_LABELS[row.J1_TELECOMMUTE_DAYS],
            commute_freq = COMMUTE_FREQ_LABELS[row.J1_COMMUTE_FREQ],
            school_freq = SCHOOL_FREQ_LABELS[row.SCHOOL_FREQ],
            school_type = SCHOOL_TYPE_LABELS[row.SCHOOL_TYPE],
            home_type = HOME_TYPE_LABELS[row.HOME_TYPE],
            home_ownership = HOME_OWNERSHIP_LABELS[row.HOME_OWNERSHIP],
            household_size = row.HHSIZE,
            gender = GENDER_LABELS[row.GENDER],
            race_ethnicity = RACEETHNICITY_LABELS[row.RACEETHNICITY],
            trips = trips
        )
        profiles.append(profile)

    return profiles

def _arrival_minutes(arrival_time: str) -> int:
    hour, minute = arrival_time.split(":")
    return int(hour) * 60 + int(minute)

EXCLUDED_FIELDS = {"archetype", "attributes", "trips"}
EV_FUEL_TYPES = {"Electric", "Plug-in Hybrid"}

def profiles_to_df(profiles: list[Profile]) -> pd.DataFrame:
    rows: list[dict] = []
    for profile in profiles:
        dict_ = profile.model_dump(mode = "json")
        row = {key: value for key, value in dict_.items() if key not in EXCLUDED_FIELDS}

        distances = [trip.distance for trip in profile.trips]
        dest_activities = [trip.dest_activity for trip in profile.trips if trip.dest_activity is not None]

        row["total_distance"] = float(sum(distances))
        row["num_trips"] = len(profile.trips)
        row["dominant_dest_activity"] = Counter(dest_activities).most_common(1)[0][0] if dest_activities else None

        for activity in set(dest_activities):
            mins = [_arrival_minutes(trip.arrival_time) for trip in profile.trips if trip.dest_activity == activity]
            row[f"mean_arrival_{activity}"] = float(sum(mins) / len(mins))

        row["has_ev"] = any(trip.vehicle and trip.vehicle.fuel_type in EV_FUEL_TYPES for trip in profile.trips)
        rows.append(row)
    
    return pd.DataFrame(rows)

def load_centroids(path: str) -> dict[int, tuple[float, float]]:
    with open(path) as file:
        centroids_json = json.load(file)
    return {int(taz_id): (lon, lat) for taz_id, (lon, lat) in centroids_json.items()}

if __name__ == "__main__":

    person_df = pd.read_csv("data/person.csv")
    household_df = pd.read_csv("data/household.csv")
    vehicle_df = pd.read_csv("data/vehicle.csv")
    trip_df = pd.read_csv("data/trip.csv")
    merged_df = pd.merge(household_df, person_df, on = "HOUSEHOLD_ID")

    tpb_centroids = load_centroids("artifacts/tpb_taz_centroids.json")
    bmc_centroids = load_centroids("artifacts/bmc_taz_centroids.json")

    profiles = build_profiles(merged_df, trip_df, vehicle_df, tpb_centroids, bmc_centroids)
    profiles_json = [json.loads(profile.model_dump_json()) for profile in profiles]

    Path("artifacts").mkdir(exist_ok = True)
    with open("artifacts/profiles.json", "w") as file:
        json.dump(profiles_json, file, indent = 2)

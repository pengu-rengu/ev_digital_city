from profiles import *
import json
import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

EXCLUDED_FIELDS = {"trips", "attributes"}


def get_ev_profiles(profiles: list[Profile]) -> list[Profile]:
    ev_profiles = []
    for profile in profiles:
        for trip in profile.trips:
            vehicle = trip.vehicle
            if vehicle and vehicle.fuel_type == "Electric":
                ev_profiles.append(profile)
                break
    return ev_profiles

def profile_to_features(profile: Profile) -> dict:
    dump = profile.model_dump(mode = "json")
    return {key: value for key, value in dump.items() if key not in EXCLUDED_FIELDS}


def build_feature_matrix(profiles: list[Profile]) -> np.ndarray:
    rows = [profile_to_features(profile) for profile in profiles]
    df = pd.DataFrame(rows).fillna("__none__").astype(str)
    encoder = OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1, encoded_missing_value = -1)
    encoded = encoder.fit_transform(df).astype(int)
    return encoded + 1


def filter_profiles(profiles: list[Profile], threshold: float = 0.4) -> list[Profile]:
    feature_matrix = build_feature_matrix(profiles)
    ev_profile_ids = {id(profile) for profile in get_ev_profiles(profiles)}
    labels = np.array([1 if id(profile) in ev_profile_ids else 0 for profile in profiles])

    sample_weight = compute_sample_weight("balanced", labels)
    clf = CategoricalNB().fit(feature_matrix, labels, sample_weight = sample_weight)
    probs = clf.predict_proba(feature_matrix)[:, 1]

    return [profile for profile, prob in zip(profiles, probs) if prob >= threshold]


if __name__ == "__main__":
    with open("artifacts/profiles.json") as file:
        profiles = [Profile.model_validate(profile_json) for profile_json in json.load(file)]

    kept = filter_profiles(profiles)
    print(f"Kept {len(kept)} of {len(profiles)} profiles (P(EV) >= 0.4)")

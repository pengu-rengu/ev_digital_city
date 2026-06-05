from profiles import *
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.contingency import association
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight


def filter_profiles(profiles: list[Profile], threshold: float, show_corr: bool = False) -> list[Profile]:
    df = profiles_to_df(profiles)
    labels = df.pop("has_ev").astype(int).to_numpy()

    keys = [key for key in profiles[0].model_dump(mode = "json") if key not in EXCLUDED_FIELDS]
    print(keys)
    df = df.fillna("__none__").astype(str)
    df = df[keys]

    if show_corr:
        cols = list(df.columns)
        records = []
        for i in range(len(cols)):
            for j in range(i, len(cols)):
                v = 1.0 if i == j else association(pd.crosstab(df[cols[i]], df[cols[j]]).to_numpy(), method = "cramer")
                records.append({"row": cols[i], "col": cols[j], "v": v})
                if i != j:
                    records.append({"row": cols[j], "col": cols[i], "v": v})
        corr = pd.DataFrame(records).pivot(index = "row", columns = "col", values = "v").reindex(index = cols, columns = cols)

        plt.figure(figsize = (12, 9))
        sns.heatmap(corr, cmap = "rocket_r", vmin = 0, vmax = 1, square = True, cbar_kws = {"shrink": 0.7})
        plt.title("Cramér's V: Features")
        plt.xticks(rotation = 45, ha = "right")
        plt.tight_layout()
        plt.show()

        target_corr = pd.Series(
            {col: association(pd.crosstab(df[col], labels).to_numpy(), method = "cramer") for col in df.columns}
        ).sort_values(ascending = False)

        plt.figure(figsize = (10, 5))
        sns.barplot(x = target_corr.index, y = target_corr.values, legend = False)
        plt.xticks(rotation = 45, ha = "right")
        plt.title("Cramér's V: Feature vs has_ev")
        plt.ylabel("Cramér's V")
        plt.tight_layout()
        plt.show()
    
    df.drop(["school_type", "school_freq", "employment_status", "workplace_loc", "commute_freq"], axis = 1, inplace = True)
    encoder = OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1, encoded_missing_value = -1)
    features = encoder.fit_transform(df).astype(int) + 1
    sample_weight = compute_sample_weight("balanced", labels)
    clf = CategoricalNB().fit(features, labels, sample_weight = sample_weight)
    probs = clf.predict_proba(features)[:, 1]

    return [profile for profile, prob, has_ev in zip(profiles, probs, labels) if has_ev or prob >= threshold]


if __name__ == "__main__":
    with open("artifacts/profiles.json") as file:
        profiles = [Profile.model_validate(profile_json) for profile_json in json.load(file)]

    filtered = filter_profiles(profiles, 0.5, True)
    print(f"Kept {len(filtered)} of {len(profiles)} profiles (P(EV) >= 0.5)")

    #profiles_json = [json.loads(profile.model_dump_json()) for profile in filtered]
    #with open("artifacts/filtered_profiles.json", "w") as file:
    #4    json.dump(profiles_json, file, indent = 2)

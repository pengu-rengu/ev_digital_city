import json
from pathlib import Path

import requests
from shapely.geometry import shape

TPB_URL = "https://gis.mwcog.org/wa/rest/services/RTDC/TAZ/MapServer/0"
BMC_URL = "https://gis.baltometro.org/arcgis/rest/services/Regional_Boundaries/MapServer/22"
PAGE_SIZE = 1000

def fetch_taz_features(service_url: str, id_field: str) -> list[dict]:
    query_url = f"{service_url}/query"
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "outFields": id_field,
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": str(PAGE_SIZE),
            "resultOffset": str(offset)
        }
        response = requests.get(query_url, params = params, timeout = 60)
        response.raise_for_status()
        page = response.json().get("features", [])
        features.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return features

def compute_centroids(features: list[dict], id_field: str) -> dict[int, tuple[float, float]]:
    centroids: dict[int, tuple[float, float]] = {}
    for feature in features:
        taz_id = int(feature["properties"][id_field])
        centroid = shape(feature["geometry"]).centroid
        centroids[taz_id] = (centroid.x, centroid.y)
    return centroids

def write_json(path: str, mapping: dict[int, tuple[float, float]]) -> None:
    serializable = {str(taz_id): [lon, lat] for taz_id, (lon, lat) in mapping.items()}
    with open(path, "w") as file:
        json.dump(serializable, file, indent = 2)

if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok = True)

    tpb_features = fetch_taz_features(TPB_URL, "TAZ")
    tpb_centroids = compute_centroids(tpb_features, "TAZ")
    write_json("artifacts/tpb_taz_centroids.json", tpb_centroids)
    print(f"TPB: {len(tpb_centroids)} centroids")

    bmc_features = fetch_taz_features(BMC_URL, "TAZ20")
    bmc_centroids = compute_centroids(bmc_features, "TAZ20")
    write_json("artifacts/bmc_taz_centroids.json", bmc_centroids)
    print(f"BMC: {len(bmc_centroids)} centroids")

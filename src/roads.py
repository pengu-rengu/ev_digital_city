import json
from pathlib import Path

import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from pydantic import BaseModel
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from nodes import fairfax_boundary, str_or_none

SERVICE_URL = "https://services.arcgis.com/p5v98VHDX9Atv3l7/ArcGIS/rest/services/VDOT_Posted_Speed_Limits/FeatureServer/0"
PAGE_SIZE = 2000
JURISDICTION = "Fairfax County"

class Road(BaseModel):
    speed_limit: int | None
    name: str | None
    length_miles: float
    coords: list[tuple[float, float]]

def fetch_speed_features() -> list[dict]:
    query_url = f"{SERVICE_URL}/query"
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": f"FROM_JURISDICTION='{JURISDICTION}'",
            "outFields": "OBJECTID,CAR_SPEED_LIMIT,ROUTE_COMMON_NAME,LENGTH",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "json",
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

def speed_or_none(value: object) -> int | None:
    return int(value) if value not in (None, 0) else None

def build_roads(features: list[dict]) -> list[Road]:
    roads: list[Road] = []
    for feature in features:
        attributes = feature["attributes"]
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        for path in geometry["paths"]:
            roads.append(Road(
                speed_limit = speed_or_none(attributes["CAR_SPEED_LIMIT"]),
                name = str_or_none(attributes["ROUTE_COMMON_NAME"]),
                length_miles = float(attributes["LENGTH"]),
                coords = [(point[0], point[1]) for point in path]
            ))
    return roads

def plot_roads(boundary: BaseGeometry, roads: list[Road]) -> None:
    frame = gpd.GeoDataFrame(
        {"speed_limit": [road.speed_limit for road in roads]},
        geometry = [LineString(road.coords) for road in roads],
        crs = "EPSG:4326"
    )
    fig, ax = plt.subplots(figsize = (12, 12))
    gpd.GeoSeries([boundary], crs = "EPSG:4326").boundary.plot(ax = ax, color = "black", linewidth = 1.2, zorder = 3)
    frame.plot(ax = ax, column = "speed_limit", cmap = "viridis", linewidth = 0.6, legend = True, legend_kwds = {"label": "Speed limit (mph)", "shrink": 0.5}, zorder = 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Fairfax County road speed limits")
    fig.tight_layout()
    fig.savefig("artifacts/roads_map.png", bbox_inches = "tight")

if __name__ == "__main__":
    boundary = fairfax_boundary()
    features = fetch_speed_features()
    roads = build_roads(features)
    plot_roads(boundary, roads)

    Path("artifacts").mkdir(exist_ok = True)
    with open("artifacts/roads.json", "w") as file:
        json.dump([json.loads(road.model_dump_json()) for road in roads], file, indent = 2)
    print(f"{len(roads)} roads")

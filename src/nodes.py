import pandas as pd
from pydantic import BaseModel
import json
import osmium
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Literal
from pathlib import Path
from shapely.geometry import Point
from shapely.prepared import prep
from shapely.geometry.base import BaseGeometry

PBF_PATH = "data/virginia-260608.osm.pbf" # fetched from GeoFabrik (https://download.geofabrik.de/north-america/us/virginia.html)
COUNTY_URL = "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip"
TIGER_DIR = Path("artifacts/tiger")
PROXIMITY_METERS = 100
UTM_CRS = "EPSG:26918"  # NAD83 / UTM 18N

def fetch_county_zip() -> Path:
    TIGER_DIR.mkdir(parents = True, exist_ok = True)
    path = TIGER_DIR / "county.zip"
    if path.exists():
        return path
    response = requests.get(COUNTY_URL, stream = True, timeout = 120)
    response.raise_for_status()
    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size = 1 << 16):
            file.write(chunk)
    return path

def fairfax_boundary() -> BaseGeometry:
    counties = gpd.read_file(f"zip://{fetch_county_zip()}").to_crs("EPSG:4326")
    return counties[counties["GEOID"].isin(["51059", "51600"])].geometry.union_all()

class ChargerNode(BaseModel):
    category: Literal["charger"]
    num_l1: int
    num_l2: int
    num_dc_fast: int
    facility_type: str | None
    ev_network: str | None
    pricing: str | None
    workplace_charging: bool
    coords: tuple[float, float]

class OsmNode(BaseModel):
    category: Literal["house", "office", "supermarket", "school", "gym", "mall", "restaurant", "clinic", "doctors", "pharmacy", "fast_food", "park", "retail", "bank", "post_office", "cinema", "cafe", "bar", "pub"]
    coords: tuple[float, float]
    chargers: list[ChargerNode] = []

def int_or_zero(value: float) -> int:
    return int(value) if pd.notna(value) else 0

def str_or_none(value: object) -> str | None:
    return value if pd.notna(value) else None

def build_chargers(df: pd.DataFrame, boundary: BaseGeometry) -> list[ChargerNode]:
    prepared = prep(boundary)
    renamed = df.rename(columns = {
        "EV Level1 EVSE Num": "num_l1",
        "EV Level2 EVSE Num": "num_l2",
        "EV DC Fast Count": "num_dc_fast",
        "Facility Type": "facility_type",
        "EV Network": "ev_network",
        "EV Pricing": "pricing",
        "EV Workplace Charging": "workplace_charging",
        "Latitude": "latitude",
        "Longitude": "longitude"
    })

    chargers = []
    for row in renamed.itertuples(index = False):
        if not prepared.contains(Point(row.longitude, row.latitude)):
            continue
        charger = ChargerNode(
            category = "charger",
            num_l1 = int_or_zero(row.num_l1),
            num_l2 = int_or_zero(row.num_l2),
            num_dc_fast = int_or_zero(row.num_dc_fast),
            facility_type = str_or_none(row.facility_type),
            ev_network = str_or_none(row.ev_network),
            pricing = str_or_none(row.pricing),
            workplace_charging = bool(row.workplace_charging),
            coords = (row.longitude, row.latitude)
        )
        chargers.append(charger)
    return chargers

def osm_category(tags) -> str | None:
    if tags.get("building") == "house":
        return "house"
    if "office" in tags:
        return "office"
    if "shop" in tags:
        return tags["shop"] if tags["shop"] in {"supermarket", "mall"} else "retail"
    if tags.get("amenity") in {"school", "restaurant", "fast_food", "clinic", "doctors", "pharmacy", "bank", "post_office", "cinema", "cafe", "bar", "pub"}:
        return tags["amenity"]
    if tags.get("leisure") in {"fitness_centre", "park"}:
        return "gym" if tags["leisure"] == "fitness_centre" else "park"
    return None

class OsmHandler(osmium.SimpleHandler):
    def __init__(self, boundary: BaseGeometry) -> None:
        super().__init__()
        self.prepared = prep(boundary)
        self.nodes = []

    def node(self, osm_node) -> None:
        category = osm_category(osm_node.tags)
        if category is not None and osm_node.location.valid() and self.prepared.contains(Point(osm_node.location.lon, osm_node.location.lat)):
            self.nodes.append(OsmNode(category = category, coords = (osm_node.location.lon, osm_node.location.lat)))

    def way(self, osm_way) -> None:
        category = osm_category(osm_way.tags)
        if category is None:
            return
        points = [(ref.lon, ref.lat) for ref in osm_way.nodes if ref.location.valid()]
        if not points:
            return
        lon = sum(point[0] for point in points) / len(points)
        lat = sum(point[1] for point in points) / len(points)
        if self.prepared.contains(Point(lon, lat)):
            self.nodes.append(OsmNode(category = category, coords = (lon, lat)))

def build_osm_nodes(pbf_path: str, boundary: BaseGeometry) -> list[OsmNode]:
    handler = OsmHandler(boundary)
    handler.apply_file(pbf_path, locations = True)
    return handler.nodes

def attach_chargers(osm_nodes: list[OsmNode], chargers: list[ChargerNode]) -> list[ChargerNode]:
    osm_frame = gpd.GeoDataFrame(
        {"osm_index": range(len(osm_nodes))},
        geometry = [Point(node.coords) for node in osm_nodes],
        crs = "EPSG:4326"
    ).to_crs(UTM_CRS)
    charger_frame = gpd.GeoDataFrame(
        {"charger_index": range(len(chargers))},
        geometry = [Point(charger.coords) for charger in chargers],
        crs = "EPSG:4326"
    ).to_crs(UTM_CRS)
    matched = gpd.sjoin_nearest(charger_frame, osm_frame, how = "left", max_distance = PROXIMITY_METERS).drop_duplicates("charger_index")

    independent = []
    for row in matched.itertuples(index = False):
        charger = chargers[row.charger_index]
        if pd.notna(row.osm_index):
            osm_nodes[int(row.osm_index)].chargers.append(charger)
        else:
            independent.append(charger)
    return independent

def plot_nodes(boundary: BaseGeometry, nodes: list[ChargerNode | OsmNode]) -> None:
    frame = gpd.GeoDataFrame(
        {"category": [node.category for node in nodes]},
        geometry = [Point(node.coords) for node in nodes],
        crs = "EPSG:4326"
    )
    fig, ax = plt.subplots(figsize = (12, 12))
    gpd.GeoSeries([boundary], crs = "EPSG:4326").boundary.plot(ax = ax, color = "black", linewidth = 1.2, zorder = 3)
    frame.plot(ax = ax, column = "category", categorical = True, cmap = "tab20", markersize = 6, alpha = 0.5, legend = True, legend_kwds = {"loc": "lower left", "fontsize": 8, "title": "Category"}, zorder = 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Fairfax County nodes")
    fig.tight_layout()
    fig.savefig("artifacts/nodes_map.png", bbox_inches = "tight")

if __name__ == "__main__":
    stations_df = pd.read_csv("data/alt_fuel_stations.csv", low_memory = False)
    boundary = fairfax_boundary()
    chargers = build_chargers(stations_df, boundary)
    osm_nodes = build_osm_nodes(PBF_PATH, boundary)
    independent_chargers = attach_chargers(osm_nodes, chargers)
    node_models = osm_nodes + independent_chargers

    nodes = [json.loads(node.model_dump_json()) for node in node_models]
    plot_nodes(boundary, node_models)

    Path("artifacts").mkdir(exist_ok = True)
    with open("artifacts/nodes.json", "w") as file:
        json.dump(nodes, file, indent = 2)

    

import json

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.geometry.base import BaseGeometry

from nodes import fairfax_boundary

def load_node_frame() -> gpd.GeoDataFrame:
    with open("artifacts/nodes.json") as file:
        nodes = json.load(file)
    return gpd.GeoDataFrame(
        {"category": [node["category"] for node in nodes]},
        geometry = [Point(node["coords"]) for node in nodes],
        crs = "EPSG:4326"
    )

def load_road_frame() -> gpd.GeoDataFrame:
    with open("artifacts/roads.json") as file:
        roads = json.load(file)
    return gpd.GeoDataFrame(
        {"speed_limit": [road["speed_limit"] for road in roads]},
        geometry = [LineString(road["coords"]) for road in roads],
        crs = "EPSG:4326"
    )

def plot_combined(boundary: BaseGeometry, node_frame: gpd.GeoDataFrame, road_frame: gpd.GeoDataFrame) -> None:
    fig, ax = plt.subplots(figsize = (12, 12))
    gpd.GeoSeries([boundary], crs = "EPSG:4326").boundary.plot(ax = ax, color = "black", linewidth = 1.2, zorder = 1)
    road_frame.plot(ax = ax, column = "speed_limit", cmap = "viridis", linewidth = 0.6, legend = True, legend_kwds = {"label": "Speed limit (mph)", "shrink": 0.5}, zorder = 2)
    node_frame.plot(ax = ax, column = "category", categorical = True, cmap = "tab20", markersize = 4, alpha = 0.25, legend = True, legend_kwds = {"loc": "lower left", "fontsize": 8, "title": "Category"}, zorder = 3)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Fairfax County nodes and roads")
    fig.tight_layout()
    fig.savefig("artifacts/combined_map.png", bbox_inches = "tight")

if __name__ == "__main__":
    boundary = fairfax_boundary()
    plot_combined(boundary, load_node_frame(), load_road_frame())

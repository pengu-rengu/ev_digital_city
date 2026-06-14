import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import DistanceTimeToNodeTool
from roads import Road


def vertex_index(coords: np.ndarray, coord: tuple[float, float]) -> int:
    for index, point in enumerate(coords):
        if tuple(point) == coord:
            return index
    raise AssertionError(f"coord {coord} not found")


def edge_speeds(adjacency: list[list[tuple[int, float, float]]], start: int, end: int) -> list[float]:
    return [
        speed
        for neighbor, _, speed in adjacency[start]
        if neighbor == end
    ]


def edge_speeds_between_coords(
    adjacency: list[list[tuple[int, float, float]]],
    coords: np.ndarray,
    start_coord: tuple[float, float],
    end_coord: tuple[float, float]
) -> list[float]:
    start_index = vertex_index(coords, start_coord)
    end_index = vertex_index(coords, end_coord)
    return edge_speeds(adjacency, start_index, end_index)


def has_edge_between_coords(
    adjacency: list[list[tuple[int, float, float]]],
    coords: np.ndarray,
    start_coord: tuple[float, float],
    end_coord: tuple[float, float],
    speed: float
) -> bool:
    return speed in edge_speeds_between_coords(adjacency, coords, start_coord, end_coord)


def coord_count(coords: np.ndarray, coord: tuple[float, float]) -> int:
    return sum(1 for point in coords if tuple(point) == coord)


def test_jump_speed_averages_endpoint_and_target_road_speed() -> None:
    roads = [
        Road(speed_limit = 20, coords = [(0.0, 0.0), (0.0, 0.01)]),
        Road(speed_limit = 40, coords = [(0.1, 0.0101), (0.2, 0.0101)])
    ]

    adjacency, coords = DistanceTimeToNodeTool(node_id = 1).build_graph(roads)

    assert 30.0 in edge_speeds_between_coords(adjacency, coords, (0.0, 0.01), (0.1, 0.0101))


def test_jump_speed_uses_average_for_shared_target_vertex() -> None:
    roads = [
        Road(speed_limit = 20, coords = [(0.0, 0.0), (0.0, 0.01)]),
        Road(speed_limit = 30, coords = [(0.1, 0.0101), (0.2, 0.0101)]),
        Road(speed_limit = 50, coords = [(0.1, 0.0101), (0.1, 0.2)])
    ]

    adjacency, coords = DistanceTimeToNodeTool(node_id = 1).build_graph(roads)

    assert 30.0 in edge_speeds_between_coords(adjacency, coords, (0.0, 0.01), (0.1, 0.0101))


def test_jump_speed_is_independent_of_road_order() -> None:
    roads = [
        Road(speed_limit = 20, coords = [(0.0, 0.0), (0.0, 0.01)]),
        Road(speed_limit = 30, coords = [(0.1, 0.0101), (0.2, 0.0101)]),
        Road(speed_limit = 50, coords = [(0.1, 0.0101), (0.1, 0.2)])
    ]
    reversed_roads = list(reversed(roads))

    adjacency, coords = DistanceTimeToNodeTool(node_id = 1).build_graph(roads)
    reversed_adjacency, reversed_coords = DistanceTimeToNodeTool(node_id = 1).build_graph(reversed_roads)

    speeds = edge_speeds_between_coords(adjacency, coords, (0.0, 0.01), (0.1, 0.0101))
    reversed_speeds = edge_speeds_between_coords(reversed_adjacency, reversed_coords, (0.0, 0.01), (0.1, 0.0101))

    assert 30.0 in speeds
    assert 30.0 in reversed_speeds


def test_crossing_roads_connect_at_inserted_intersection() -> None:
    roads = [
        Road(speed_limit = 20, coords = [(-1.0, 0.0), (1.0, 0.0)]),
        Road(speed_limit = 40, coords = [(0.0, -1.0), (0.0, 1.0)])
    ]

    adjacency, coords = DistanceTimeToNodeTool(node_id = 1).build_graph(roads)

    assert coord_count(coords, (0.0, 0.0)) == 1
    assert has_edge_between_coords(adjacency, coords, (0.0, 0.0), (-1.0, 0.0), 20.0)
    assert has_edge_between_coords(adjacency, coords, (0.0, 0.0), (1.0, 0.0), 20.0)
    assert has_edge_between_coords(adjacency, coords, (0.0, 0.0), (0.0, -1.0), 40.0)
    assert has_edge_between_coords(adjacency, coords, (0.0, 0.0), (0.0, 1.0), 40.0)


def test_existing_coordinate_intersections_share_one_vertex() -> None:
    roads = [
        Road(speed_limit = 20, coords = [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]),
        Road(speed_limit = 40, coords = [(0.0, -1.0), (0.0, 0.0), (0.0, 1.0)])
    ]

    adjacency, coords = DistanceTimeToNodeTool(node_id = 1).build_graph(roads)

    assert coord_count(coords, (0.0, 0.0)) == 1
    assert has_edge_between_coords(adjacency, coords, (0.0, 0.0), (-1.0, 0.0), 20.0)
    assert has_edge_between_coords(adjacency, coords, (0.0, 0.0), (0.0, -1.0), 40.0)


def test_overlapping_roads_do_not_crash() -> None:
    roads = [
        Road(speed_limit = 20, coords = [(0.0, 0.0), (2.0, 0.0)]),
        Road(speed_limit = 40, coords = [(1.0, 0.0), (3.0, 0.0)])
    ]

    adjacency, coords = DistanceTimeToNodeTool(node_id = 1).build_graph(roads)

    assert len(adjacency) == len(coords)

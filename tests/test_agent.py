import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import Agent, AgentAction, AppendToScheduleTool, DistanceTimeToNodeTool, FinishTool, NodeBlock, Schedule, SearchNodesTool, SetStartTimeTool, TravelBlock, run_agent
from nodes import ChargerNode, OsmNode
from profiles import Archetype, Attributes, MobilityLevel
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


def test_format_shows_attached_and_standalone_chargers() -> None:
    attached = ChargerNode(
        category = "charger",
        num_l1 = 0,
        num_l2 = 4,
        num_dc_fast = 2,
        facility_type = None,
        ev_network = "ChargePoint",
        pricing = "$0.30/kWh",
        workplace_charging = False,
        coords = (0.0, 0.0)
    )
    store = OsmNode(category = "supermarket", metadata = {"name": "Whole Foods"}, coords = (0.0, 0.0), chargers = [attached])
    standalone = ChargerNode(
        category = "charger",
        num_l1 = 1,
        num_l2 = 0,
        num_dc_fast = 0,
        facility_type = None,
        ev_network = None,
        pricing = None,
        workplace_charging = True,
        coords = (0.1, 0.1)
    )

    text = SearchNodesTool.format_result([store, standalone])

    assert f"Node ID: {store.id}" in text
    assert "Category: supermarket" in text
    assert "Name: Whole Foods" in text
    assert "Charging Station:" in text
    assert "Level 2 Ports: 4" in text
    assert "Network: ChargePoint" in text
    assert f"Node ID: {standalone.id}" in text
    assert "Level 1 Ports: 1" in text
    assert "Workplace Charging: yes" in text


def test_format_result_distance_time() -> None:
    text = DistanceTimeToNodeTool(node_id = 1).format_result((3.25, 12.0))

    assert "3.2 miles" in text
    assert "00:12" in text


def test_schedule_format() -> None:
    gym = OsmNode(category = "gym", metadata = {}, coords = (0.0, 0.0))
    schedule = Schedule(
        start_time = 480,
        blocks = [
            TravelBlock(start_time = 480, end_time = 495),
            NodeBlock(node_id = gym.id, start_time = 495, end_time = 525)
        ]
    )

    text = schedule.format([gym])

    assert "Start Time: 08:00" in text
    assert "Travel: 08:00 - 08:15" in text
    assert "gym: 08:15 - 08:45" in text


class FakeParsed:
    def __init__(self, action: object) -> None:
        self.output = []
        self.output_text = ""
        self.output_parsed = AgentAction(thought = "", action = action)


class FakeResponses:
    def __init__(self, actions: list[object]) -> None:
        self.actions = actions
        self.calls = 0

    def parse(self, model: str, input: list[dict], text_format: type) -> FakeParsed:
        action = self.actions[self.calls]
        self.calls += 1
        return FakeParsed(action)


class FakeClient:
    def __init__(self, actions: list[object]) -> None:
        self.responses = FakeResponses(actions)


def test_run_agent_drives_tools() -> None:
    home = OsmNode(category = "house", metadata = {}, coords = (0.0, 0.0))
    agent = Agent(
        persona = "A commuter.",
        archetype = Archetype.FLEXIBLE_COMMUTER,
        attributes = Attributes(is_caregiver = False, mobility_level = MobilityLevel.MODERATE, work_arrangement = None, schedule_irregular = False),
        schedule = Schedule(start_time = None, blocks = []),
        context = [],
        home_node_id = home.id
    )
    client = FakeClient([SetStartTimeTool(start_hh_mm = "08:00"), FinishTool()])

    result = run_agent(agent, [home], [], client)

    assert result.schedule.start_time == 480
    assert client.responses.calls == 2


def test_search_caps_at_20() -> None:
    origin = OsmNode(category = "house", metadata = {}, coords = (0.0, 0.0))
    offices = [OsmNode(category = "office", metadata = {}, coords = (0.0, 0.0001 * (index + 1))) for index in range(30)]
    nodes = [origin] + offices

    found = SearchNodesTool(radius = 10, categories = ["office"]).run(origin.id, nodes)

    assert len(found) == 20
    assert all(node.category == "office" for node in found)


def test_append_rounds_travel_times() -> None:
    home = OsmNode(category = "house", metadata = {}, coords = (0.0, 0.0))
    dest = OsmNode(category = "office", metadata = {}, coords = (0.0, 0.01))
    roads = [Road(speed_limit = 25, coords = [(0.0, 0.0), (0.0, 0.01)])]
    agent = Agent(
        persona = "A commuter.",
        archetype = Archetype.FLEXIBLE_COMMUTER,
        attributes = Attributes(is_caregiver = False, mobility_level = MobilityLevel.MODERATE, work_arrangement = None, schedule_irregular = False),
        schedule = Schedule(start_time = 480, blocks = []),
        context = [],
        home_node_id = home.id
    )

    schedule = AppendToScheduleTool(node_id = dest.id, dwell_time = 10).run(agent, [home, dest], roads)

    for block in schedule.blocks:
        assert isinstance(block.start_time, int)
        assert isinstance(block.end_time, int)

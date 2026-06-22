import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import Agent, NodeBlock, Schedule, TravelBlock
from nodes import ChargerNode, OsmNode
from profiles import Archetype
from simulation import ChargeResolution, ListChargeStopsTool, ReadjustChargeTool, WaitInQueueTool, build_simulation_events, first_contention, resolve_contentions, sessions_for


def charger_node(coords: tuple[float, float], num_l2: int) -> ChargerNode:
    return ChargerNode(
        category = "charger",
        num_l1 = 0,
        num_l2 = num_l2,
        num_dc_fast = 0,
        facility_type = None,
        ev_network = None,
        pricing = None,
        workplace_charging = False,
        coords = coords
    )


def charging_agent(home_node_id: int, blocks: list[NodeBlock]) -> Agent:
    return Agent(
        persona = "A commuter.",
        archetype = Archetype.FLEXIBLE_COMMUTER,
        schedules = [Schedule(start_time = 470, blocks = blocks)],
        contexts = [[]],
        home_node_id = home_node_id,
        battery_kwh = 60.0,
        start_soc_kwh = 60.0,
        soc_kwh = 60.0
    )


class FakeParsed:
    def __init__(self, action: object) -> None:
        self.output = []
        self.output_text = ""
        self.output_parsed = ChargeResolution(thought = "", action = action)


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


def test_detection_flags_latest_arrival() -> None:
    node = OsmNode(category = "office", metadata = {}, coords = (0.0, 0.01), chargers = [charger_node((0.0, 0.01), 1)])
    early = charging_agent(node.id, [NodeBlock(node_id = node.id, start_time = 470, end_time = 700, charge_level = "L2", charge_start_time = 480, charge_duration = 60)])
    late = charging_agent(node.id, [NodeBlock(node_id = node.id, start_time = 470, end_time = 700, charge_level = "L2", charge_start_time = 510, charge_duration = 60)])

    contended = first_contention(sessions_for([early, late], 0), [node])

    assert contended is not None
    assert contended.agent_index == 1


def test_detection_clears_with_enough_ports() -> None:
    node = OsmNode(category = "office", metadata = {}, coords = (0.0, 0.01), chargers = [charger_node((0.0, 0.01), 2)])
    early = charging_agent(node.id, [NodeBlock(node_id = node.id, start_time = 470, end_time = 700, charge_level = "L2", charge_start_time = 480, charge_duration = 60)])
    late = charging_agent(node.id, [NodeBlock(node_id = node.id, start_time = 470, end_time = 700, charge_level = "L2", charge_start_time = 510, charge_duration = 60)])

    assert first_contention(sessions_for([early, late], 0), [node]) is None


def test_queue_resolution_shifts_start() -> None:
    node = OsmNode(category = "office", metadata = {}, coords = (0.0, 0.01), chargers = [charger_node((0.0, 0.01), 1)])
    early = charging_agent(node.id, [NodeBlock(node_id = node.id, start_time = 470, end_time = 700, charge_level = "L2", charge_start_time = 480, charge_duration = 60)])
    late = charging_agent(node.id, [NodeBlock(node_id = node.id, start_time = 470, end_time = 700, charge_level = "L2", charge_start_time = 510, charge_duration = 60)])
    client = FakeClient([WaitInQueueTool()])

    events = resolve_contentions([early, late], [node], client, 0)

    assert len(events) == 1
    assert events[0].resolution == "queued"
    assert len(events[0].reasoning) == 1
    assert late.schedules[0].blocks[0].charge_start_time == 540


def test_queue_no_fit_relocates() -> None:
    node = OsmNode(category = "office", metadata = {}, coords = (0.0, 0.01), chargers = [charger_node((0.0, 0.01), 1)])
    other = OsmNode(category = "gym", metadata = {}, coords = (0.0, 0.02), chargers = [charger_node((0.0, 0.02), 1)])
    early = charging_agent(node.id, [NodeBlock(node_id = node.id, start_time = 470, end_time = 700, charge_level = "L2", charge_start_time = 480, charge_duration = 60)])
    late = charging_agent(node.id, [
        NodeBlock(node_id = node.id, start_time = 470, end_time = 575, charge_level = "L2", charge_start_time = 510, charge_duration = 60),
        NodeBlock(node_id = other.id, start_time = 600, end_time = 800)
    ])
    client = FakeClient([WaitInQueueTool(), ReadjustChargeTool(node_id = other.id, charge_level = "L2", charge_start_hh_mm = "10:30")])

    events = resolve_contentions([early, late], [node, other], client, 0)

    assert len(events) == 1
    assert events[0].resolution == "relocated"
    assert len(events[0].reasoning) == 2
    assert late.schedules[0].blocks[0].charge_level is None
    assert late.schedules[0].blocks[1].charge_level == "L2"
    assert late.schedules[0].blocks[1].charge_start_time == 630


def test_list_charge_stops_then_relocate() -> None:
    node = OsmNode(category = "office", metadata = {}, coords = (0.0, 0.01), chargers = [charger_node((0.0, 0.01), 1)])
    other = OsmNode(category = "gym", metadata = {}, coords = (0.0, 0.02), chargers = [charger_node((0.0, 0.02), 1)])
    early = charging_agent(node.id, [NodeBlock(node_id = node.id, start_time = 470, end_time = 700, charge_level = "L2", charge_start_time = 480, charge_duration = 60)])
    late = charging_agent(node.id, [
        NodeBlock(node_id = node.id, start_time = 470, end_time = 575, charge_level = "L2", charge_start_time = 510, charge_duration = 60),
        NodeBlock(node_id = other.id, start_time = 600, end_time = 800)
    ])
    client = FakeClient([ListChargeStopsTool(), ReadjustChargeTool(node_id = other.id, charge_level = "L2", charge_start_hh_mm = "10:30")])

    events = resolve_contentions([early, late], [node, other], client, 0)

    assert len(events) == 1
    assert events[0].resolution == "relocated"
    assert len(events[0].reasoning) == 2
    assert any(f"Node ID: {other.id}" in message["content"] for message in late.contexts[0] if message["role"] == "user")
    assert sum(1 for message in late.contexts[0] if "Charger contention:" in message["content"]) == 1


def test_build_simulation_events() -> None:
    node = OsmNode(category = "office", metadata = {}, coords = (0.0, 0.01), chargers = [charger_node((0.0, 0.01), 1)])
    agent = charging_agent(99, [
        TravelBlock(start_time = 480, end_time = 486, distance = 2.0),
        NodeBlock(node_id = node.id, start_time = 486, end_time = 660, charge_level = "L2", charge_start_time = 540, charge_duration = 60)
    ])

    events = build_simulation_events([agent], 0)
    status_at = {event.time: event.statuses[0].status for event in events}

    assert status_at[480] == "traveling"
    assert status_at[486] == "at_node"
    assert status_at[540] == "charging"
    assert status_at[600] == "at_node"
    assert status_at[660] == "home"

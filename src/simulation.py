import json
import os
from typing import Literal
import ollama
from pydantic import BaseModel
from agent import Agent, NodeBlock, Schedule, CHARGE_POWER_KW, level_ports, DAY_NAMES, replay_soc
from llm import call_llm
from nodes import OsmNode, ChargerNode
from profiles import hhmm_to_mins, mins_to_hhmm

class ChargeSession(BaseModel):
    agent_index: int
    block: NodeBlock

class ContentionEvent(BaseModel):
    agent_index: int
    node_id: int
    level: str
    resolution: Literal["queued", "relocated", "gave_up"]
    detail: str
    reasoning: list[str]

class AgentStatus(BaseModel):
    agent_index: int
    status: Literal["home", "traveling", "at_node", "charging"]
    node_id: int | None

class SimulationEvent(BaseModel):
    time: int
    statuses: list[AgentStatus]

class ListChargeStopsTool(BaseModel):
    tool: Literal["list_charge_stops"] = "list_charge_stops"

    def run(self, schedule: Schedule, nodes: list[OsmNode | ChargerNode]) -> str:
        node_by_id = {node.id: node for node in nodes}
        text = ""
        for block in schedule.blocks:
            if not isinstance(block, NodeBlock):
                continue
            node = node_by_id[block.node_id]
            levels = [level for level in CHARGE_POWER_KW if level_ports(node, level) > 0]
            if not levels:
                continue
            text += f"Node ID: {node.id}\n"
            text += f"Category: {node.category}\n"
            text += f"Stop: {mins_to_hhmm(block.start_time)} - {mins_to_hhmm(block.end_time)}\n"
            for level in levels:
                text += f"  {level} Ports: {level_ports(node, level)}\n"
            text += "\n"
        return text or "None of your stops have a charger.\n"

class WaitInQueueTool(BaseModel):
    tool: Literal["wait_in_queue"] = "wait_in_queue"

    def run(self, session: ChargeSession, sessions: list[ChargeSession], nodes: list[OsmNode | ChargerNode]) -> str:
        node = next(node for node in nodes if node.id == session.block.node_id)
        capacity = level_ports(node, session.block.charge_level)
        free_at = earliest_free(session, sessions, capacity)
        if free_at + session.block.charge_duration > session.block.end_time:
            raise ValueError("Queue pushes your charge past the end of your stop; relocate instead")
        session.block.charge_start_time = free_at
        return f"You wait in a queue; charge start moved to {mins_to_hhmm(free_at)}."

class ReadjustChargeTool(BaseModel):
    tool: Literal["readjust_charge"] = "readjust_charge"
    node_id: int
    charge_level: Literal["L1", "L2", "DC"]
    charge_start_hh_mm: str

    def run(self, schedule: Schedule, session: ChargeSession, nodes: list[OsmNode | ChargerNode]) -> str:
        target = next((block for block in schedule.blocks if isinstance(block, NodeBlock) and block.node_id == self.node_id), None)
        if target is None:
            raise ValueError(f"You do not visit node {self.node_id}; choose a stop already in your schedule")

        node = next(node for node in nodes if node.id == self.node_id)
        if level_ports(node, self.charge_level) == 0:
            raise ValueError(f"Node {self.node_id} has no {self.charge_level} ports")

        duration = session.block.charge_duration
        new_start = hhmm_to_mins(self.charge_start_hh_mm)
        new_end = new_start + duration

        original_start = session.block.charge_start_time
        if new_start < original_start:
            raise ValueError(f"Readjusted charge must start at or after your original window {mins_to_hhmm(original_start)}; cannot move earlier")

        if new_start < target.start_time or new_end > target.end_time:
            raise ValueError(f"Charge {mins_to_hhmm(new_start)}-{mins_to_hhmm(new_end)} must fit your stop {mins_to_hhmm(target.start_time)}-{mins_to_hhmm(target.end_time)}")

        session.block.charge_level = None
        session.block.charge_start_time = None
        session.block.charge_duration = None
        target.charge_level = self.charge_level
        target.charge_start_time = new_start
        target.charge_duration = duration
        return f"Charge moved to node {self.node_id} at {mins_to_hhmm(new_start)}."

class GiveUpTool(BaseModel):
    tool: Literal["give_up"] = "give_up"

    def run(self, session: ChargeSession) -> str:
        node_id = session.block.node_id
        level = session.block.charge_level
        session.block.charge_level = None
        session.block.charge_start_time = None
        return f"You give up charging; dropped {level} charge at node {node_id}."

class ChargeResolution(BaseModel):
    thought: str
    action: ListChargeStopsTool | WaitInQueueTool | ReadjustChargeTool | GiveUpTool

def session_end(block: NodeBlock) -> int:
    return block.charge_start_time + block.charge_duration

def sessions_for(agents: list[Agent], day_index: int) -> list[ChargeSession]:
    return [
        ChargeSession(agent_index = index, block = block)
        for index, agent in enumerate(agents)
        for block in agent.schedules[day_index].blocks
        if isinstance(block, NodeBlock) and block.charge_level
    ]

def first_contention(sessions: list[ChargeSession], nodes: list[OsmNode | ChargerNode]) -> ChargeSession | None:
    node_by_id = {node.id: node for node in nodes}
    groups = {}
    for session in sessions:
        groups.setdefault((session.block.node_id, session.block.charge_level), []).append(session)
    for (node_id, level), group in groups.items():
        capacity = level_ports(node_by_id[node_id], level)
        ordered = sorted(group, key = lambda session: (session.block.charge_start_time, session.agent_index))
        for index, session in enumerate(ordered):
            start = session.block.charge_start_time
            active = sum(1 for earlier in ordered[:index] if session_end(earlier.block) > start)
            if active >= capacity:
                return session
    return None

def earliest_free(session: ChargeSession, sessions: list[ChargeSession], capacity: int) -> int:
    start = session.block.charge_start_time
    blockers = sorted(
        session_end(other.block)
        for other in sessions
        if other is not session
        and other.block.node_id == session.block.node_id
        and other.block.charge_level == session.block.charge_level
        and (other.block.charge_start_time, other.agent_index) < (start, session.agent_index)
        and session_end(other.block) > start
    )
    if len(blockers) < capacity:
        return start
    return blockers[len(blockers) - capacity]

def resolve_contentions(agents: list[Agent], nodes: list[OsmNode | ChargerNode], client: ollama.Client, day_index: int) -> list[ContentionEvent]:
    events = []
    prompted = {}
    reasoning_traces = {}
    while True:
        sessions = sessions_for(agents, day_index)
        contended = first_contention(sessions, nodes)
        if contended is None:
            return events

        agent = agents[contended.agent_index]
        context = agent.contexts[day_index]
        schedule = agent.schedules[day_index]
        node_id = contended.block.node_id
        level = contended.block.charge_level
        charge_start = contended.block.charge_start_time
        if prompted.get(contended.agent_index) != (node_id, level, charge_start):
            charge_end = session_end(contended.block)
            context.append({
                "role": "user",
                "content": (
                    f"Charger contention: all {level} ports at node {node_id} are taken during your charge window "
                    f"{mins_to_hhmm(charge_start)}-{mins_to_hhmm(charge_end)}. "
                    "Wait in queue (delay your charge start until a port frees; it must still finish before you leave the stop), "
                    "or use list_charge_stops to see which of your stops have a charger, then readjust your charge to one of them, "
                    "or give up charging at this stop if neither works."
                )
            })
            prompted[contended.agent_index] = (node_id, level, charge_start)
        parsed = call_llm(client, context, schema = ChargeResolution)
        print(parsed, end = "\n\n\n")
        context.append({"role": "assistant", "content": parsed.model_dump_json()})
        action = parsed.action
        reasoning_traces.setdefault(contended.agent_index, []).append(parsed.thought)

        try:
            if isinstance(action, ListChargeStopsTool):
                output = action.run(schedule, nodes)
            elif isinstance(action, WaitInQueueTool):
                output = action.run(contended, sessions, nodes)
                events.append(ContentionEvent(
                    agent_index = contended.agent_index,
                    node_id = node_id,
                    level = level,
                    resolution = "queued",
                    detail = output,
                    reasoning = reasoning_traces.pop(contended.agent_index)
                ))
            elif isinstance(action, ReadjustChargeTool):
                output = action.run(schedule, contended, nodes)
                events.append(ContentionEvent(
                    agent_index = contended.agent_index,
                    node_id = action.node_id,
                    level = action.charge_level,
                    resolution = "relocated",
                    detail = output,
                    reasoning = reasoning_traces.pop(contended.agent_index)
                ))
            else:
                output = action.run(contended)
                events.append(ContentionEvent(
                    agent_index = contended.agent_index,
                    node_id = node_id,
                    level = level,
                    resolution = "gave_up",
                    detail = output,
                    reasoning = reasoning_traces.pop(contended.agent_index)
                ))
        except ValueError as error:
            output = str(error)

        print(output)
        context.append({"role": "user", "content": output})

def agent_status(agent: Agent, day_index: int, time: int) -> tuple[str, int | None]:
    for block in agent.schedules[day_index].blocks:
        if block.start_time <= time < block.end_time:
            if isinstance(block, NodeBlock):
                if block.charge_level and block.charge_start_time <= time < block.charge_start_time + block.charge_duration:
                    return "charging", block.node_id
                return "at_node", block.node_id
            return "traveling", None
    return "home", agent.home_node_id

def build_simulation_events(agents: list[Agent], day_index: int) -> list[SimulationEvent]:
    starts = [block.start_time for agent in agents for block in agent.schedules[day_index].blocks]
    ends = [block.end_time for agent in agents for block in agent.schedules[day_index].blocks]
    events = []
    for time in range(min(starts), max(ends) + 1, 3):
        statuses = [
            AgentStatus(agent_index = index, status = status, node_id = node_id)
            for index, agent in enumerate(agents)
            for status, node_id in [agent_status(agent, day_index, time)]
        ]
        events.append(SimulationEvent(time = time, statuses = statuses))
    return events

if __name__ == "__main__":
    client = ollama.Client()

    agent_path = "artifacts/agents.json"
    simulation_path = "artifacts/simulation_logs.json"

    with open(agent_path) as file:
        agent_data = json.load(file)
    with open("artifacts/nodes.json") as file:
        nodes = [ChargerNode.model_validate(node) if node["category"] == "charger" else OsmNode.model_validate(node) for node in json.load(file)]

    agent_index = agent_data["agent_index"]
    start = agent_data["simulation_index"]
    agents = [Agent.model_validate(agent) for agent in agent_data["agents"]]

    if os.path.exists(simulation_path):
        with open(simulation_path) as file:
            data = json.load(file)
        start = data["simulation_index"]
        days = data["days"]
    else:
        days = []

    print(f"Processing simulation days {start}..{len(DAY_NAMES) - 1} for {len(agents)} agents")
    for day_index in range(start, len(DAY_NAMES)):
        print(f"[{day_index + 1}/{len(DAY_NAMES)}] simulating {DAY_NAMES[day_index]}...")
        start_socs = [agent.soc_kwh for agent in agents]
        contention_events = resolve_contentions(agents, nodes, client, day_index)
        for agent, start_soc in zip(agents, start_socs):
            agent.soc_kwh = replay_soc(agent.schedules[day_index], start_soc, agent.battery_kwh)
        simulation_events = build_simulation_events(agents, day_index)

        days.append({
            "day": DAY_NAMES[day_index],
            "contention_events": [event.model_dump() for event in contention_events],
            "simulation_events": [event.model_dump() for event in simulation_events]
        })
        next_index = day_index + 1

        with open(simulation_path, "w") as file:
            json.dump({
                "simulation_index": next_index,
                "days": days
            }, file, indent = 2)
        with open(agent_path, "w") as file:
            json.dump({
                "agent_index": agent_index,
                "simulation_index": next_index,
                "agents": [json.loads(agent.model_dump_json()) for agent in agents]
            }, file, indent = 2)
        print(f"saved simulation day {DAY_NAMES[day_index]}, next simulation index {next_index}")

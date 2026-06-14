import json
import dotenv
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel
from agent import Agent, NodeBlock, Schedule, CHARGE_DURATION, level_ports
from nodes import OsmNode, ChargerNode
from profiles import hhmm_to_mins, mins_to_hhmm

class ChargeSession(BaseModel):
    agent_index: int
    block: NodeBlock

class ContentionEvent(BaseModel):
    agent_index: int
    node_id: int
    level: str
    resolution: Literal["queued", "relocated"]
    detail: str

class ListChargeStopsTool(BaseModel):
    tool: Literal["list_charge_stops"] = "list_charge_stops"

    def run(self, agent: Agent, nodes: list[OsmNode | ChargerNode]) -> str:
        node_by_id = {node.id: node for node in nodes}
        text = ""
        for block in agent.schedule.blocks:
            if not isinstance(block, NodeBlock):
                continue
            node = node_by_id[block.node_id]
            levels = [level for level in CHARGE_DURATION if level_ports(node, level) > 0]
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
        if free_at + CHARGE_DURATION[session.block.charge_level] > session.block.end_time:
            raise ValueError("Queue pushes your charge past the end of your stop; relocate instead")
        session.block.charge_start_time = free_at
        return f"You wait in a queue; charge start moved to {mins_to_hhmm(free_at)}."

class ReadjustChargeTool(BaseModel):
    tool: Literal["readjust_charge"] = "readjust_charge"
    node_id: int
    charge_level: Literal["L1", "L2", "DC"]
    charge_start_hh_mm: str

    def run(self, agent: Agent, session: ChargeSession, nodes: list[OsmNode | ChargerNode]) -> str:
        target = next((block for block in agent.schedule.blocks if isinstance(block, NodeBlock) and block.node_id == self.node_id), None)
        if target is None:
            raise ValueError(f"You do not visit node {self.node_id}; choose a stop already in your schedule")

        node = next(node for node in nodes if node.id == self.node_id)
        if level_ports(node, self.charge_level) == 0:
            raise ValueError(f"Node {self.node_id} has no {self.charge_level} ports")

        new_start = hhmm_to_mins(self.charge_start_hh_mm)
        new_end = new_start + CHARGE_DURATION[self.charge_level]

        if new_start < target.start_time or new_end > target.end_time:
            raise ValueError(f"Charge {mins_to_hhmm(new_start)}-{mins_to_hhmm(new_end)} must fit your stop {mins_to_hhmm(target.start_time)}-{mins_to_hhmm(target.end_time)}")

        session.block.charge_level = None
        session.block.charge_start_time = None
        target.charge_level = self.charge_level
        target.charge_start_time = new_start
        return f"Charge moved to node {self.node_id} at {mins_to_hhmm(new_start)}."

class ChargeResolution(BaseModel):
    thought: str
    action: ListChargeStopsTool | WaitInQueueTool | ReadjustChargeTool

def session_end(block: NodeBlock) -> int:
    return block.charge_start_time + CHARGE_DURATION[block.charge_level]

def sessions_for(agents: list[Agent]) -> list[ChargeSession]:
    return [
        ChargeSession(agent_index = index, block = block)
        for index, agent in enumerate(agents)
        for block in agent.schedule.blocks
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

def resolve_contention(agents: list[Agent], nodes: list[OsmNode | ChargerNode], client: OpenAI, max_rounds: int = 100) -> list[ContentionEvent]:
    events = []
    prompted = {}
    for _ in range(max_rounds):
        sessions = sessions_for(agents)
        contended = first_contention(sessions, nodes)
        if contended is None:
            return events

        agent = agents[contended.agent_index]
        node_id = contended.block.node_id
        level = contended.block.charge_level
        charge_start = contended.block.charge_start_time
        if prompted.get(contended.agent_index) != (node_id, level, charge_start):
            charge_end = session_end(contended.block)
            agent.context.append({
                "role": "user",
                "content": (
                    f"Charger contention: all {level} ports at node {node_id} are taken during your charge window "
                    f"{mins_to_hhmm(charge_start)}-{mins_to_hhmm(charge_end)}. "
                    "Wait in queue (delay your charge start until a port frees; it must still finish before you leave the stop), "
                    "or use list_charge_stops to see which of your stops have a charger, then readjust your charge to one of them."
                )
            })
            prompted[contended.agent_index] = (node_id, level, charge_start)
        response = client.responses.parse(
            model = "gpt-5.4-mini",
            input = agent.context,
            text_format = ChargeResolution
        )
        print(response.output_text, end = "\n\n\n")
        agent.context.append({"role": "assistant", "content": response.output_text})
        action = response.output_parsed.action

        try:
            if isinstance(action, ListChargeStopsTool):
                output = action.run(agent, nodes)
            elif isinstance(action, WaitInQueueTool):
                output = action.run(contended, sessions, nodes)
                events.append(ContentionEvent(
                    agent_index = contended.agent_index,
                    node_id = node_id,
                    level = level,
                    resolution = "queued", 
                    detail = output
                ))
            else:
                output = action.run(agent, contended, nodes)
                events.append(ContentionEvent(
                    agent_index = contended.agent_index,
                    node_id = action.node_id,
                    level = action.charge_level,
                    resolution = "relocated",
                    detail = output
                ))
        except ValueError as error:
            output = str(error)

        print(output)
        agent.context.append({"role": "user", "content": output})

    raise RuntimeError(f"charger contention unresolved within {max_rounds} rounds")

def simulate(agents: list[Agent], nodes: list[OsmNode | ChargerNode], client: OpenAI, max_rounds: int = 100) -> list[ContentionEvent]:
    return resolve_contention(agents, nodes, client, max_rounds)

if __name__ == "__main__":
    dotenv.load_dotenv(override = True)
    client = OpenAI()

    with open("artifacts/agents.json") as file:
        agents = [Agent.model_validate(agent) for agent in json.load(file)]
    with open("artifacts/nodes.json") as file:
        nodes = [ChargerNode.model_validate(node) if node["category"] == "charger" else OsmNode.model_validate(node) for node in json.load(file)]

    real_agent = agents[0]
    gym_node_id = 312
    mock_a = Agent(
        persona = "Mock driver already charging at the gym.",
        archetype = real_agent.archetype,
        attributes = real_agent.attributes,
        schedule = Schedule(start_time = 340, blocks = [NodeBlock(node_id = gym_node_id, start_time = 340, end_time = 410, charge_level = "L2", charge_start_time = 346)]),
        context = [],
        home_node_id = real_agent.home_node_id
    )
    mock_b = Agent(
        persona = "Mock driver already charging at the gym.",
        archetype = real_agent.archetype,
        attributes = real_agent.attributes,
        schedule = Schedule(start_time = 340, blocks = [NodeBlock(node_id = gym_node_id, start_time = 340, end_time = 470, charge_level = "L2", charge_start_time = 400)]),
        context = [],
        home_node_id = real_agent.home_node_id
    )

    contending_agents = [real_agent, mock_a, mock_b]
    events = simulate(contending_agents, nodes, client)

    for event in events:
        print(event)
    print()
    for agent in contending_agents:
        print(agent.schedule.format(nodes))
        print(agent.context)

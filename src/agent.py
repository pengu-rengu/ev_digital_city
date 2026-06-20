import dotenv
import json
import math
import heapq
import random
import numpy as np
from typing import Literal
from openai import OpenAI
from personas import PersonaArtifact, format_profile
from pydantic import BaseModel
from profiles import Archetype, Attributes, Profile, hhmm_to_mins, mins_to_hhmm
from nodes import OsmNode, ChargerNode
from roads import Road
from shapely.geometry import LineString, MultiPoint, Point
from shapely.geometry.base import BaseGeometry

CHARGE_POWER_KW: dict[str, float] = {"L1": 1.4, "L2": 7.2, "DC": 50.0}
PORT_FIELD: dict[str, str] = {"L1": "num_l1", "L2": "num_l2", "DC": "num_dc_fast"}
CONSUMPTION_KWH_PER_MI: float = 0.30
BATTERY_KWH_MEAN: float = 65.0
BATTERY_KWH_STD: float = 18.0
BATTERY_KWH_MIN: float = 24.0
BATTERY_KWH_MAX: float = 100.0

def level_ports(node: OsmNode | ChargerNode, level: str) -> int:
    if isinstance(node, ChargerNode):
        return getattr(node, PORT_FIELD[level])
    return sum(getattr(charger, PORT_FIELD[level]) for charger in node.chargers)

class NodeBlock(BaseModel):
    node_id: int
    start_time: int
    end_time: int
    charge_level: Literal["L1", "L2", "DC"] | None = None
    charge_start_time: int | None = None
    charge_duration: int | None = None

class TravelBlock(BaseModel):
    start_time: int
    end_time: int
    distance: float

class Schedule(BaseModel):
    start_time: int | None
    blocks: list[TravelBlock | NodeBlock]

    def format(self, nodes: list[OsmNode | ChargerNode]) -> str:
        text = ""
        if self.start_time is None:
            text += "Start Time: no start time\n"
        else:
            text += f"Start Time: {mins_to_hhmm(self.start_time)}\n"
        for block in self.blocks:
            start = mins_to_hhmm(block.start_time)
            end = mins_to_hhmm(block.end_time)
            if isinstance(block, NodeBlock):
                category = next(node.category for node in nodes if node.id == block.node_id)
                if block.charge_level:
                    charge_end = block.charge_start_time + block.charge_duration
                    added = CHARGE_POWER_KW[block.charge_level] * block.charge_duration / 60
                    text += f"{category}: {start} - {end} (charge {block.charge_level} {mins_to_hhmm(block.charge_start_time)}-{mins_to_hhmm(charge_end)} +{added:.1f} kWh)\n"
                else:
                    text += f"{category}: {start} - {end}\n"
            else:
                text += f"Travel: {start} - {end}\n"
        return text

class Agent(BaseModel):
    persona: str
    profile: Profile | None = None
    archetype: Archetype
    attributes: Attributes
    day_type: Literal["weekday", "weekend"]
    schedule: Schedule
    context: list[dict]
    home_node_id: int
    battery_kwh: float
    start_soc_kwh: float
    soc_kwh: float

def agent_from_persona_artifact(persona_artifact: PersonaArtifact, home_node_id: int, day_type: Literal["weekday", "weekend"]) -> Agent:
    battery_kwh = float(np.clip(np.random.normal(BATTERY_KWH_MEAN, BATTERY_KWH_STD), BATTERY_KWH_MIN, BATTERY_KWH_MAX))
    start_soc_kwh = battery_kwh * float(np.random.triangular(0.1, 0.3, 0.9))
    return Agent(
        persona = persona_artifact.personas[persona_artifact.best_index].persona,
        profile = persona_artifact.target_profile,
        archetype = persona_artifact.target_profile.archetype,
        attributes = persona_artifact.target_profile.attributes,
        day_type = day_type,
        schedule = Schedule(
            start_time = None,
            blocks = []
        ),
        home_node_id = home_node_id,
        context = [],
        battery_kwh = battery_kwh,
        start_soc_kwh = start_soc_kwh,
        soc_kwh = start_soc_kwh
    )

def haversine_miles(origin: tuple[float, float], dest: tuple[float, float]) -> float:
    lon1, lat1 = origin
    lon2, lat2 = dest
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    inner = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(inner))

def coords_for_node_id(node_id: int, nodes: list[OsmNode | ChargerNode]) -> tuple[float, float] | None:
    return next((node.coords for node in nodes if node.id == node_id), None)

class SearchNodesTool(BaseModel):
    tool: Literal["search_nodes"] = "search_nodes"
    radius: float
    categories: list[str]

    def run(self, origin_node_id: int, nodes: list[OsmNode | ChargerNode]) -> list[OsmNode | ChargerNode]:
        origin_coords = coords_for_node_id(origin_node_id, nodes)
        if origin_coords is None:
            raise ValueError(f"node id {origin_node_id} not found")
        matches = [
            node for node in nodes
            if node.category in self.categories and haversine_miles(origin_coords, node.coords) <= self.radius
        ]
        random.shuffle(matches)
        return matches[:10]

    @staticmethod
    def format_charger(charger: ChargerNode) -> str:
        text = ""
        if charger.num_l1:
            text += f"  Level 1 Ports: {charger.num_l1}\n"
        if charger.num_l2:
            text += f"  Level 2 Ports: {charger.num_l2}\n"
        if charger.num_dc_fast:
            text += f"  DC Fast Ports: {charger.num_dc_fast}\n"
        if charger.ev_network:
            text += f"  Network: {charger.ev_network}\n"
        if charger.pricing:
            text += f"  Pricing: {charger.pricing}\n"
        if charger.workplace_charging:
            text += "  Workplace Charging: yes\n"
        return text

    @staticmethod
    def format_result(nodes: list[OsmNode | ChargerNode]) -> str:
        text = ""
        for node in nodes:
            text += f"Node ID: {node.id}\n"
            text += f"Category: {node.category}\n"
            if isinstance(node, ChargerNode):
                text += SearchNodesTool.format_charger(node)
            else:
                name = node.metadata.get("name")
                if name:
                    text += f"Name: {name}\n"
                for charger in node.chargers:
                    text += "Charging Station:\n"
                    text += SearchNodesTool.format_charger(charger)
            text += "\n"
        return text


class DistanceTimeToNodeTool(BaseModel):
    tool: Literal["distance_time_to_node"] = "distance_time_to_node"
    node_id: int

    def point_intersections(self, geometry: BaseGeometry) -> list[Point]:
        if geometry.is_empty:
            return []
        if isinstance(geometry, Point):
            return [geometry]
        if isinstance(geometry, MultiPoint):
            return list(geometry.geoms)
        return []

    def augmented_road_coords(self, roads: list[Road]) -> list[list[tuple[float, float]]]:
        lines = [LineString(road.coords) for road in roads]
        extra_coords_by_road = [set() for _ in roads]

        for first_index in range(len(roads)):
            for second_index in range(first_index + 1, len(roads)):
                intersection = lines[first_index].intersection(lines[second_index])
                for point in self.point_intersections(intersection):
                    coord = (point.x, point.y)
                    extra_coords_by_road[first_index].add(coord)
                    extra_coords_by_road[second_index].add(coord)

        return [
            sorted(
                set(road.coords) | extra_coords_by_road[road_index],
                key = lambda coord: lines[road_index].project(Point(coord))
            )
            for road_index, road in enumerate(roads)
        ]

    def nearest_vertex(self, coord: tuple[float, float], coords: np.ndarray, exclude: frozenset = frozenset()) -> int | None:
        distances = [math.inf if index in exclude else haversine_miles(coord, (point[0], point[1])) for index, point in enumerate(coords)]
        nearest = int(np.argmin(distances))
        return nearest if math.isfinite(distances[nearest]) else None

    def build_graph(self, roads: list[Road]) -> tuple[list[list[tuple[int, float, float]]], np.ndarray]:
        vertex_ids = {}
        vertex_coords = []
        vertex_road_speeds = []
        adjacency = []
        road_coords_by_road = self.augmented_road_coords(roads)

        for road, road_coords in zip(roads, road_coords_by_road):
            road_speed = float(road.speed_limit)
            previous = None
            road_vertex_ids = set()
            for coord in road_coords:
                key = (coord[0], coord[1])
                if key not in vertex_ids:
                    vertex_ids[key] = len(vertex_coords)
                    vertex_coords.append(key)
                    vertex_road_speeds.append([])
                    adjacency.append([])
                current = vertex_ids[key]
                road_vertex_ids.add(current)
                if previous is not None and previous != current:
                    miles = haversine_miles(vertex_coords[previous], vertex_coords[current])
                    adjacency[previous].append((current, miles, road_speed))
                    adjacency[current].append((previous, miles, road_speed))
                previous = current
            for vertex_id in road_vertex_ids:
                vertex_road_speeds[vertex_id].append(road_speed)

        coords = np.array(vertex_coords)
        for road, road_coords in zip(roads, road_coords_by_road):
            ids = frozenset(vertex_ids[(coord[0], coord[1])] for coord in road_coords)
            endpoints = {vertex_ids[(road_coords[0][0], road_coords[0][1])], vertex_ids[(road_coords[-1][0], road_coords[-1][1])]}
            for endpoint in endpoints:
                target = self.nearest_vertex(vertex_coords[endpoint], coords, exclude = ids)
                if target is None:
                    continue
                miles = haversine_miles(vertex_coords[endpoint], vertex_coords[target])
                target_speed = sum(vertex_road_speeds[target]) / len(vertex_road_speeds[target])
                speed = (float(road.speed_limit) + target_speed) / 2
                adjacency[endpoint].append((target, miles, speed))
                adjacency[target].append((endpoint, miles, speed))
        return adjacency, coords

    def astar(self, adjacency: list[list[tuple[int, float, float]]], coords: np.ndarray, start: int, end: int) -> tuple[float, float] | None:
        end_coord = (coords[end][0], coords[end][1])
        best_distance = {start: 0.0}
        predecessor = {start: None}
        heap = [(haversine_miles((coords[start][0], coords[start][1]), end_coord), 0.0, start)]
        visited = set()
        while heap:
            _, distance, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == end:
                minutes = 0.0
                cursor = end
                while predecessor[cursor] is not None:
                    previous, miles, speed = predecessor[cursor]
                    minutes += miles / speed * 60
                    cursor = previous
                
                return distance, minutes
            
            for neighbor, miles, speed in adjacency[node]:
                if neighbor in visited:
                    continue
                tentative = distance + miles
                if neighbor not in best_distance or tentative < best_distance[neighbor]:
                    best_distance[neighbor] = tentative
                    predecessor[neighbor] = (node, miles, speed)
                    heuristic = haversine_miles((coords[neighbor][0], coords[neighbor][1]), end_coord)
                    heapq.heappush(heap, (tentative + heuristic, tentative, neighbor))
        return None

    def run(self, origin_node_id: int, nodes: list[OsmNode | ChargerNode], roads: list[Road]) -> tuple[float, float] | None:
        dest_coords = coords_for_node_id(self.node_id, nodes)
        if dest_coords is None:
            raise ValueError(f"node id {self.node_id} not found")
        origin_coords = coords_for_node_id(origin_node_id, nodes)
        if origin_coords is None:
            raise ValueError(f"node id {origin_node_id} not found")

        adjacency, coords = self.build_graph(roads)
        start = self.nearest_vertex(origin_coords, coords)
        end = self.nearest_vertex(dest_coords, coords)
        result = self.astar(adjacency, coords, start, end)

        if not result:
            raise ValueError(f"Distance and time to node {self.node_id} calculation failed")

        return result

    def format_result(self, dist_time_result: tuple[float, float]) -> str:
        dist, minutes = dist_time_result
        return f"Distance: {dist:.1f} miles\nTravel Time: {mins_to_hhmm(minutes)}\n"
    
class SetStartTimeTool(BaseModel):
    tool: Literal["set_start_time"] = "set_start_time"
    start_hh_mm: str

    def run(self, schedule: Schedule) -> Schedule:
        new_schedule = schedule.model_copy(deep = True)

        if new_schedule.start_time:
            raise ValueError("Schedule already has start time")

        new_schedule.start_time = hhmm_to_mins(self.start_hh_mm)
        if new_schedule.start_time > 1440 or new_schedule.start_time <= 0:
            raise ValueError(f"Invalid schedule start time: {self.start_hh_mm}")
        
        return new_schedule

class ResetScheduleTool(BaseModel):
    tool: Literal["reset_schedule"] = "reset_schedule"

    def run(self, schedule: Schedule) -> Schedule:
        new_schedule = schedule.model_copy(deep = True)
        new_schedule.start_time = None
        new_schedule.blocks = []
        return new_schedule

class AppendToScheduleTool(BaseModel):
    tool: Literal["append_to_schedule"] = "append_to_schedule"
    node_id: int
    dwell_time: int
    charge_level: Literal["L1", "L2", "DC"] | None = None
    charge_start_hh_mm: str | None = None
    charge_duration: int | None = None

    def run(self, agent: Agent, nodes: list[OsmNode | ChargerNode], roads: list[Road]) -> Schedule:
        new_schedule = agent.schedule.model_copy(deep = True)

        if not new_schedule.start_time:
            raise ValueError(f"Schedule must have start time. Use set_start_time tool")

        if new_schedule.blocks:

            t = new_schedule.blocks.pop().start_time
            node_to_node_dist, node_to_node_minutes = DistanceTimeToNodeTool(
                node_id = self.node_id
            ).run(new_schedule.blocks[-1].node_id, nodes, roads)
            node_to_node_time = math.ceil(node_to_node_minutes)

            new_schedule.blocks.append(TravelBlock(
                start_time = t,
                end_time = t + node_to_node_time,
                distance = node_to_node_dist
            ))
            t += node_to_node_time
            trip_distance = node_to_node_dist
        else:
            t = new_schedule.start_time
            home_to_node_dist, home_to_node_minutes = DistanceTimeToNodeTool(
                node_id = self.node_id
            ).run(agent.home_node_id, nodes, roads)
            home_to_node_time = math.ceil(home_to_node_minutes)


            new_schedule.blocks.append(TravelBlock(
                start_time = t,
                end_time = t + home_to_node_time,
                distance = home_to_node_dist
            ))
            t += home_to_node_time
            trip_distance = home_to_node_dist

        arrival_soc = agent.soc_kwh - trip_distance * CONSUMPTION_KWH_PER_MI
        if arrival_soc <= 0:
            raise ValueError(
                f"Battery would reach 0 driving to node {self.node_id} "
                f"({trip_distance:.1f} miles, {trip_distance * CONSUMPTION_KWH_PER_MI:.1f} kWh needed, "
                f"{agent.soc_kwh:.1f} kWh on hand). Charge earlier or pick a closer stop."
            )

        node_block = NodeBlock(
            node_id = self.node_id,
            start_time = t,
            end_time = t + self.dwell_time
        )
        new_schedule.blocks.append(node_block)

        new_soc = arrival_soc
        if self.charge_level is not None:
            node = next(node for node in nodes if node.id == self.node_id)
            if level_ports(node, self.charge_level) == 0:
                raise ValueError(f"Node {self.node_id} has no {self.charge_level} ports")

            if self.charge_duration is None or self.charge_duration <= 0:
                raise ValueError("Charging requires a positive charge_duration in minutes")

            charge_start = hhmm_to_mins(self.charge_start_hh_mm)
            charge_end = charge_start + self.charge_duration

            if charge_start < node_block.start_time or charge_end > node_block.end_time:
                raise ValueError(
                    f"Charge {mins_to_hhmm(charge_start)}-{mins_to_hhmm(charge_end)} "
                    f"must fit your stop {mins_to_hhmm(node_block.start_time)}-{mins_to_hhmm(node_block.end_time)}"
                )
            node_block.charge_level = self.charge_level
            node_block.charge_start_time = charge_start
            node_block.charge_duration = self.charge_duration

            added = CHARGE_POWER_KW[self.charge_level] * self.charge_duration / 60
            new_soc = min(agent.battery_kwh, arrival_soc + added)

        t += self.dwell_time
        node_to_home_dist, node_to_home_minutes = DistanceTimeToNodeTool(
            node_id = agent.home_node_id
        ).run(self.node_id, nodes, roads)
        node_to_home_time = math.ceil(node_to_home_minutes)

        new_schedule.blocks.append(TravelBlock(
            start_time = t,
            end_time = t + node_to_home_time,
            distance = node_to_home_dist
        ))

        agent.soc_kwh = new_soc
        return new_schedule

class FinishTool(BaseModel):
    tool: Literal["finish"] = "finish"

class AgentAction(BaseModel):
    thought: str
    action: SearchNodesTool | DistanceTimeToNodeTool | SetStartTimeTool | AppendToScheduleTool | ResetScheduleTool | FinishTool

def current_node_id(agent: Agent) -> int:
    for block in reversed(agent.schedule.blocks):
        if isinstance(block, NodeBlock):
            return block.node_id
    return agent.home_node_id

def build_system_prompt(agent: Agent, use_profile: bool = False) -> str:
    work_arrangement = agent.attributes.work_arrangement.value if agent.attributes.work_arrangement else "Unknown"
    if use_profile:
        intro = f"Profile:\n{format_profile(agent.profile)}\n"
    else:
        intro = f"Persona:\n{agent.persona}"
    return f"""You are role-playing as the following person, planning where you go on a typical {agent.day_type}.

{intro}

Archetype: {agent.archetype.value}
Caregiver: {agent.attributes.is_caregiver}
Mobility: {agent.attributes.mobility_level.value}
Work Arrangement: {work_arrangement}
Irregular Schedule: {agent.attributes.schedule_irregular}

Today is a {agent.day_type}. Your routine differs between weekdays and weekends, so plan the day that fits a typical {agent.day_type} for this person.
You start and end the day at your home (node id {agent.home_node_id}).
Node categories: house, office, supermarket, school, gym, mall, restaurant, clinic, doctors, pharmacy, fast_food, park, retail, bank, post_office, cinema, cafe, bar, pub.

You drive an electric vehicle with a {agent.battery_kwh:.0f} kWh battery, starting the day at {agent.start_soc_kwh:.0f} kWh. Driving uses about {CONSUMPTION_KWH_PER_MI} kWh per mile, so your battery drains as you travel. You cannot charge at home; any charging happens at an away-from-home stop that has a charging station.
Charge by passing charge_level (L1, L2, or DC), charge_start_hh_mm, and charge_duration (minutes — you choose how long) to append_to_schedule; you can only charge at a node that has a "Charging Station".
Charging adds power x time: L1 ~ 1.4 kW, L2 ~ 7.2 kW, DC ~ 50 kW (so 60 min of DC adds ~50 kWh, capped at your battery size). The full charge window must fit inside that stop's dwell time, so make the dwell long enough.
You may charge zero or more times during the day. The only rule: your battery must never reach 0. Your remaining battery is shown after each stop.

Respond with exactly ONE action per turn — never multiple. Build a realistic daily schedule:
1. Call set_start_time first with the time (HH:MM) you leave home.
2. Use search_nodes and distance_time_to_node to explore options from your current location.
3. Call append_to_schedule for each stop with a dwell time in minutes; travel legs are inserted automatically. Charge at a stop using charge_level, charge_start_hh_mm, and charge_duration.
4. When your day is complete and you can make it back home without running out, call finish.

If you make a mistake, call reset_schedule to clear your schedule and start over from set_start_time."""

def run_agent(agent: Agent, nodes: list[OsmNode | ChargerNode], roads: list[Road], client: OpenAI, max_turns: int = 20) -> Agent:

    system_prompt = build_system_prompt(agent)
    print(system_prompt)

    agent.context = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Plan your full day."}
    ]

    for _ in range(max_turns):
        response = client.responses.parse(
            model = "gpt-5.4-mini",
            input = agent.context,
            text_format = AgentAction
        )
        print(response.output_text, end = "\n\n\n")
        agent.context.append({"role": "assistant", "content": response.output_text})
        action = response.output_parsed.action
        if isinstance(action, FinishTool):
            home_trip = next((block for block in reversed(agent.schedule.blocks) if isinstance(block, TravelBlock)), None)
            home_trip_energy = home_trip.distance * CONSUMPTION_KWH_PER_MI if home_trip else 0.0
            if agent.soc_kwh - home_trip_energy <= 0:
                agent.context.append({"role": "user", "content": (
                    f"You cannot make it home: {home_trip_energy:.1f} kWh needed for the drive home "
                    f"but only {agent.soc_kwh:.1f} kWh on hand. Add or extend a charge before finishing."
                )})
                continue
            agent.soc_kwh -= home_trip_energy
            break

        try:
            if isinstance(action, SearchNodesTool):
                output = SearchNodesTool.format_result(action.run(current_node_id(agent), nodes))
            elif isinstance(action, DistanceTimeToNodeTool):
                output = action.format_result(action.run(current_node_id(agent), nodes, roads))
            elif isinstance(action, (SetStartTimeTool, ResetScheduleTool)):
                agent.schedule = action.run(agent.schedule)
                if isinstance(action, ResetScheduleTool):
                    agent.soc_kwh = agent.start_soc_kwh
                    agent.context = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Plan your full day."}
                    ]
                output = agent.schedule.format(nodes)
                output += f"Battery: {agent.soc_kwh:.1f} / {agent.battery_kwh:.1f} kWh ({agent.soc_kwh / agent.battery_kwh:.0%})\n"
            else:
                agent.schedule = action.run(agent, nodes, roads)
                output = agent.schedule.format(nodes)
                output += f"Battery: {agent.soc_kwh:.1f} / {agent.battery_kwh:.1f} kWh ({agent.soc_kwh / agent.battery_kwh:.0%})\n"
        except ValueError as error:
            output = str(error)

        print(output)
        agent.context.append({"role": "user", "content": output})

    return agent

if __name__ == "__main__":
    dotenv.load_dotenv(override = True)
    client = OpenAI()

    with open("artifacts/personas.json") as file:
        artifact = PersonaArtifact.model_validate(json.load(file)[1])
    
    with open("artifacts/nodes.json") as file:
        nodes = [ChargerNode.model_validate(node) if node["category"] == "charger" else OsmNode.model_validate(node) for node in json.load(file)]
    with open("artifacts/roads.json") as file:
        roads = [Road.model_validate(road) for road in json.load(file)]

    home_node_id = next(node.id for node in nodes if node.category == "house")
    agents = [
        run_agent(agent_from_persona_artifact(artifact, home_node_id, day_type), nodes, roads, client)
        for day_type in ("weekday", "weekend")
    ]

    #for message in agents[0].context:
    #    print(message, end = "\n\n")

    with open("artifacts/agents.json", "w") as file:
        json.dump([json.loads(agent.model_dump_json()) for agent in agents], file, indent = 2)
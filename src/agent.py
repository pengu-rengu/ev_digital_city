import dotenv
import json
import math
import heapq
import random
import numpy as np
from typing import Literal
from openai import OpenAI
from personas import PersonaArtifact
from pydantic import BaseModel
from profiles import Archetype, Attributes, hhmm_to_mins, mins_to_hhmm
from nodes import OsmNode, ChargerNode
from roads import Road
from shapely.geometry import LineString, MultiPoint, Point
from shapely.geometry.base import BaseGeometry

class NodeBlock(BaseModel):
    node_id: int
    start_time: int
    end_time: int

class TravelBlock(BaseModel):
    start_time: int
    end_time: int

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
                text += f"{category}: {start} - {end}\n"
            else:
                text += f"Travel: {start} - {end}\n"
        return text

class Agent(BaseModel):
    persona: str
    archetype: Archetype
    attributes: Attributes
    schedule: Schedule
    context: list[dict]
    home_node_id: int

def agent_from_persona_artifact(persona_artifact: PersonaArtifact, home_node_id: int) -> Agent:
    return Agent(
        persona = persona_artifact.best_persona,
        archetype = persona_artifact.target_profile.archetype,
        attributes = persona_artifact.target_profile.attributes,
        schedule = Schedule(
            start_time = None,
            blocks = []
        ),
        home_node_id = home_node_id,
        context = []
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
        return matches[:20]

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

class AppendToScheduleTool(BaseModel):
    tool: Literal["append_to_schedule"] = "append_to_schedule"
    node_id: int
    dwell_time: int

    def run(self, agent: Agent, nodes: list[OsmNode | ChargerNode], roads: list[Road]) -> Schedule:
        new_schedule = agent.schedule.model_copy(deep = True)

        if not new_schedule.start_time:
            raise ValueError(f"Schedule must have start time. Use set_start_time tool")

        if new_schedule.blocks:
            
            t = new_schedule.blocks.pop().start_time
            node_to_node_time = round(DistanceTimeToNodeTool(
                node_id = self.node_id
            ).run(new_schedule.blocks[-1].node_id, nodes, roads)[1])

            new_schedule.blocks.append(TravelBlock(
                start_time = t,
                end_time = t + node_to_node_time
            ))
            t += node_to_node_time
        else:
            t = new_schedule.start_time
            home_to_node_time = round(DistanceTimeToNodeTool(
                node_id = self.node_id
            ).run(agent.home_node_id, nodes, roads)[1])

            
            new_schedule.blocks.append(TravelBlock(
                start_time = t, 
                end_time = t + home_to_node_time
            ))
            t += home_to_node_time
        
        new_schedule.blocks.append(NodeBlock(
            node_id = self.node_id, 
            start_time = t, 
            end_time = t + self.dwell_time
        ))

        t += self.dwell_time
        node_to_home_time = round(DistanceTimeToNodeTool(
            node_id = agent.home_node_id
        ).run(self.node_id, nodes, roads)[1])

        new_schedule.blocks.append(TravelBlock(
            start_time = t,
            end_time = t + node_to_home_time
        ))

        return new_schedule

class FinishTool(BaseModel):
    tool: Literal["finish"] = "finish"

class AgentAction(BaseModel):
    thought: str
    action: SearchNodesTool | DistanceTimeToNodeTool | SetStartTimeTool | AppendToScheduleTool | FinishTool

def current_node_id(agent: Agent) -> int:
    for block in reversed(agent.schedule.blocks):
        if isinstance(block, NodeBlock):
            return block.node_id
    return agent.home_node_id

def build_system_prompt(agent: Agent) -> str:
    work_arrangement = agent.attributes.work_arrangement.value if agent.attributes.work_arrangement else "Unknown"
    return f"""You are role-playing as the following person, planning where you go on one typical day.

Persona:
{agent.persona}

Archetype: {agent.archetype.value}
Caregiver: {agent.attributes.is_caregiver}
Mobility: {agent.attributes.mobility_level.value}
Work Arrangement: {work_arrangement}
Irregular Schedule: {agent.attributes.schedule_irregular}

You start and end the day at your home (node id {agent.home_node_id}).
Node categories: house, office, supermarket, school, gym, mall, restaurant, clinic, doctors, pharmacy, fast_food, park, retail, bank, post_office, cinema, cafe, bar, pub.

Respond with exactly ONE action per turn — never multiple. Build a realistic daily schedule:
1. Call set_start_time first with the time (HH:MM) you leave home.
2. Use search_nodes and distance_time_to_node to explore options from your current location.
3. Call append_to_schedule for each stop with a dwell time in minutes; travel legs are inserted automatically.
4. When your day is complete and you are back home, call finish."""

def run_agent(agent: Agent, nodes: list[OsmNode | ChargerNode], roads: list[Road], client: OpenAI, max_turns: int = 20) -> Agent:
    agent.context = [
        {"role": "system", "content": build_system_prompt(agent)},
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
            break

        try:
            if isinstance(action, SearchNodesTool):
                output = SearchNodesTool.format_result(action.run(current_node_id(agent), nodes))
            elif isinstance(action, DistanceTimeToNodeTool):
                output = action.format_result(action.run(current_node_id(agent), nodes, roads))
            elif isinstance(action, SetStartTimeTool):
                agent.schedule = action.run(agent.schedule)
                output = agent.schedule.format(nodes)
            else:
                agent.schedule = action.run(agent, nodes, roads)
                output = agent.schedule.format(nodes)
        except ValueError as error:
            output = str(error)

        print(output)
        agent.context.append({"role": "user", "content": output})

    return agent

if __name__ == "__main__":
    dotenv.load_dotenv(override = True)
    client = OpenAI()

    with open("artifacts/personas.json") as file:
        artifact = PersonaArtifact.model_validate(json.load(file)[0])
    with open("artifacts/nodes.json") as file:
        nodes = [ChargerNode.model_validate(node) if node["category"] == "charger" else OsmNode.model_validate(node) for node in json.load(file)]
    with open("artifacts/roads.json") as file:
        roads = [Road.model_validate(road) for road in json.load(file)]

    home_node_id = next(node.id for node in nodes if node.category == "house")
    agent = run_agent(agent_from_persona_artifact(artifact, home_node_id), nodes, roads, client)

    for message in agent.context:
        print(message, end = "\n\n")
import dotenv
import json
import math
import heapq
import numpy as np
from personas import PersonaArtifact
from pydantic import BaseModel
from profiles import Archetype, Attributes, hhmm_to_mins
from nodes import OsmNode, ChargerNode
from roads import Road

class NodeBlock(BaseModel):
    node_id: int
    start_time: int
    end_time: int

class TravelBlock(BaseModel):
    start_time: int
    end_time: int

class Schedule(BaseModel):
    start_time: int
    blocks: list[TravelBlock | NodeBlock]

class Agent(BaseModel):
    persona: str
    archetype: Archetype
    attributes: Attributes
    schedule: Schedule
    start_node_id: int

def agent_from_persona_artifact(persona_artifact: PersonaArtifact, start_node_id: int) -> Agent:
    return Agent(
        persona = persona_artifact.best_persona,
        archetype = persona_artifact.target_profile.archetype,
        attributes = persona_artifact.target_profile.attributes,
        schedule = Schedule(
            start_time = 0.0,
            blocks = []
        ),
        start_node_id = start_node_id
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
    radius: float
    categories: list[str]

    def run(self, origin_node_id: int, nodes: list[OsmNode | ChargerNode]) -> list[OsmNode | ChargerNode]:
        origin_coords = coords_for_node_id(origin_node_id, nodes)
        if origin_coords is None:
            raise ValueError(f"node id {origin_node_id} not found")
        return [
            node for node in nodes
            if node.category in self.categories and haversine_miles(origin_coords, node.coords) <= self.radius
        ]

class DistanceTimeToNodeTool(BaseModel):
    node_id: int

    def nearest_vertex(self, coord: tuple[float, float], coords: np.ndarray, exclude: frozenset = frozenset()) -> int | None:
        distances = [math.inf if index in exclude else haversine_miles(coord, (point[0], point[1])) for index, point in enumerate(coords)]
        nearest = int(np.argmin(distances))
        return nearest if math.isfinite(distances[nearest]) else None

    def build_graph(self, roads: list[Road]) -> tuple[list[list[tuple[int, float, float]]], np.ndarray]:
        vertex_ids = {}
        vertex_coords = []
        vertex_speed = []
        adjacency = []

        for road in roads:
            previous = None
            for coord in road.coords:
                key = (coord[0], coord[1])
                if key not in vertex_ids:
                    vertex_ids[key] = len(vertex_coords)
                    vertex_coords.append(key)
                    vertex_speed.append(road.speed_limit)
                    adjacency.append([])
                current = vertex_ids[key]
                if previous is not None and previous != current:
                    miles = haversine_miles(vertex_coords[previous], vertex_coords[current])
                    adjacency[previous].append((current, miles, road.speed_limit))
                    adjacency[current].append((previous, miles, road.speed_limit))
                previous = current

        coords = np.array(vertex_coords)
        for road in roads:
            ids = frozenset(vertex_ids[(coord[0], coord[1])] for coord in road.coords)
            endpoints = {vertex_ids[(road.coords[0][0], road.coords[0][1])], vertex_ids[(road.coords[-1][0], road.coords[-1][1])]}
            for endpoint in endpoints:
                target = self.nearest_vertex(vertex_coords[endpoint], coords, exclude = ids)
                if target is None:
                    continue
                miles = haversine_miles(vertex_coords[endpoint], vertex_coords[target])
                speed = vertex_speed[target]
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
    
class SetStartTimeTool(BaseModel):
    start_hh_mm: str

    def run(self, schedule: Schedule) -> Schedule:
        new_schedule = schedule.model_copy(deep = True)
        new_schedule.start_time = hhmm_to_mins(self.start_hh_mm)
        if new_schedule.start_time > 1440 or new_schedule.start_time < 0:
            raise ValueError(f"Invalid schedule start time: {self.start_hh_mm}")
        
        return new_schedule

class AppendToScheduleTool(BaseModel):
    node_id: int
    dwell_time: int

    def run(self, agent: Agent, nodes: list[OsmNode | ChargerNode], roads: list[Road]) -> Schedule:
        new_schedule = agent.schedule.model_copy(deep = True)

        if new_schedule.blocks:
            pass
        else:
            pass

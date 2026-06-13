import dotenv
import json
import math
from personas import PersonaArtifact
from pydantic import BaseModel
from profiles import Archetype, Attributes
from nodes import OsmNode, ChargerNode

class Agent(BaseModel):
    persona: str
    archetype: Archetype
    attributes: Attributes

def agent_from_persona_artifact(persona_artifact: PersonaArtifact) -> Agent:
    return Agent(
        persona = persona_artifact.best_persona,
        archetype = persona_artifact.target_profile.archetype,
        attributes = persona_artifact.target_profile.attributes
    )

class SearchNodesTool(BaseModel):
    radius: float
    categories: list[str]

    def haversine_miles(self, origin: tuple[float, float], dest: tuple[float, float]) -> float:
        lon1, lat1 = origin
        lon2, lat2 = dest
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        inner = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        return 3958.8 * 2 * math.asin(math.sqrt(inner))

    def run(self, origin: tuple[float, float], nodes: list[OsmNode | ChargerNode]) -> list[OsmNode | ChargerNode]:
        return [
            node for node in nodes
            if node.category in self.categories and self.haversine_miles(origin, node.coords) <= self.radius
        ]

import dotenv
import json
from personas import PersonaArtifact
from pydantic import BaseModel
from profiles import Archetype, Attributes

class Agent(BaseModel):
    persona: str
    archetype: Archetype
    attributes: Attributes

def agent_from_persona_artifact(persona_artifact: PersonaArtifact) -> Agent:
    pass

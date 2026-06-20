import dotenv
import json
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel
from profiles import Profile, Archetype
from persona_data import DISPOSITIONS, SCENARIOS

class ScenarioResponse(BaseModel):
    reasoning: str
    action: Literal["A", "B", "C"]

class RankingResult(BaseModel):
    reasoning: str
    ranking: list[int]

class ScenarioRanking(BaseModel):
    scenario: str
    reasoning: str
    ranking: list[int]

class PersonaCandidate(BaseModel):
    persona: str
    disposition: str
    actions: list[ScenarioResponse]
    score: int

class PersonaArtifact(BaseModel):
    personas: list[PersonaCandidate]
    best_index: int
    target_profile: Profile
    rankings: list[ScenarioRanking]

def format_profile(profile: Profile) -> str:
    text = f"Age Group: {profile.age_group}\n"
    if profile.household_income:
        text += f"Household Income: {profile.household_income}\n"
    if profile.employment_status:
        text += f"Employment Status: {profile.employment_status}\n"
    if profile.student_status:
        text += f"Student Status: {profile.student_status}\n"
    if profile.household_size:
        text += f"Household Size: {profile.household_size}\n"
    if profile.home_type:
        text += f"Home Type: {profile.home_type}\n"
    if profile.home_ownership:
        text += f"Home Ownership: {profile.home_ownership}\n"
    if profile.workplace_ev_charging is not None:
        text += f"Workplace EV Charging: {'available' if profile.workplace_ev_charging else 'not available'}\n"
    if profile.telecommute_days:
        text += f"Telecommute Days: {profile.telecommute_days}\n"
    if profile.commute_freq:
        text += f"Commute Frequency: {profile.commute_freq}\n"

    text += "\nTrips:\n\n"

    for trip in profile.trips:
        text += f"Departure Time: {trip.departure_time}\n"
        text += f"Arrival time: {trip.arrival_time}\n"
        text += f"Origin Activity: {trip.origin_activity}\n"
        text += f"Destination Activity: {trip.dest_activity}\n"
        text += f"Travel Distance: {trip.distance} miles\n"
        text += f"Travel Time: {trip.travel_time} minutes\n\n"

    return text

def generate_persona(profile: Profile, client: OpenAI, disposition: str) -> str:
    system_prompt = """You are a social scientist building grounded, realistic personas of real people for a travel-behavior study. You will be given a Profile with demographics and a list of trips taken on a single day, along with a disposition: a short description of the person's general behavioral attitude — how they travel, run their day, and charge their EV. Your persona must plausibly fit the Profile and the disposition, and make the person's routine, day-to-day variation, and EV charging habits inferable.

DON'T:
- Do not restate the trip list, time-stamped itineraries, or hour-by-hour schedules.
- Do not restate age group, household income, employment status, student status, or household size word for word.
- Do not use the broad activity categories in the trip list, such as Work, Meal, or Shopping.
- Do not lock the person into one fixed daily schedule; the trip list is one day out of many.
- Do not assume home charging: this person has no home charging and only charges away from home.
- Do not state exact kWh, exact battery percentages at exact times, or a fixed charging schedule.
- Do not use "probably", "likely", or "suggests". State tendencies as a fact.

DO:
- Do give general demographics that help infer the person's trip schedule.
- Do give time windows, such as early morning or late afternoon, and usual activities, each with a real-world reason.
- Do go into detail for the activities instead of leaving them as broad categories
- Do separate anchor activities, such as work, or school pickup from loose discretionary activities, such as gym, errands, meals.
- Do give frequencies, tendencies, and conditional day-to-day variation, not single occurrences.
- Do describe activities across weeks or months instead of being locked in to one day.
- Do give preferences and trade-offs so the person's choices can be inferred in new situations.
- Do convey away-from-home EV charging behavior: where they plug in, how low they let the battery get, fast DC vs slower port, how they react to a busy charger, and price sensitivity.

EXAMPLES:

Bad: He leaves the house at 6:30 am
Good: He gets up very early in the morning to avoid traffic, except on days where he works remote

Bad: She has a pick-up trip at 3:30 pm
Good: She usually picks up her kids from a nearby elementary school in the mid-afternoon, around when school typically ends. Every Wednesday, however, she instead pick ups her kids from soccer practice in the early evening.

Bad: He takes a break from work and goes to a fast food restaurant around 12:30 pm
Good: He does not like cooking at home and values convenience, especially for lunch, although he is still willing to cook for special occasions.

Bad: Some days he skips the gym
Good: Some days he skips the lifing weights when work runs long

Bad: She goes to the gym in the morning
Good: She goes swimming at the gym's indoor pool in the morning during the weekdays. However, occasionally the pool is closed, in which case she stays at home.

Bad: He charges to 80% at 2:15 pm
Good: He tops up while already stopped for an errand and avoids letting the battery get too low

Bad: She charges every day right after work
Good: On long-driving days she seeks out a fast charger; most days she charges while running errands

Bad: He eats a meal for dinner after shopping
Good: He brings his family to the grocery store every Saturday, and afterwards they eat dinner at a Chinese restaurant.

Bad: He goes to a healthcare location in the evening
Good: He picks up his medicine from the pharmacy on the first of every month
"""

    profile_str = format_profile(profile)
    user_prompt = f"""Profile:
{profile_str}

Disposition:
{disposition}

Generate one persona from this profile and disposition"""

    response = client.responses.create(
        model = "gpt-5.4-mini",
        input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    print(response.output_text, end = "\n\n\n")
    return response.output_text

def answer_scenario(persona: str, scenario: str, client: OpenAI) -> ScenarioResponse:
    system_prompt = f"""You are the following persona. Stay fully in character as this person.

Persona:
{persona}

You drive an electric vehicle and have no way to charge at home, so any charging you do happens away from home during the day.

You will be given a situation about your travel, charging, or daily life, with options A, B, and C. Choose exactly one option as your action and explain why in character. Narrate the reasoning in first-person, not as a third party."""
    user_prompt = scenario

    response = client.responses.parse(
        model = "gpt-5.4-mini",
        input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text_format = ScenarioResponse
    )
    print(response.output_parsed, end = "\n\n\n")
    return response.output_parsed

def rank_personas(profile: Profile, scenario: str, responses: list[ScenarioResponse], client: OpenAI) -> RankingResult:
    profile_str = format_profile(profile)
    responses_text = "\n".join(
        f"Person {index + 1} chose option {response.action}.\nPerson {index + 1} reasoning:\n{response.reasoning}\n"
        for index, response in enumerate(responses)
    )
    system_prompt = """You are a travel-behavior and EV charging expert. Several people each describe how they would handle the same situation. Rank them from most to least realistic for the actual person described in the Profile, given who they are and how they live.

Assume no one has access to home charging. Treat any action or reasoning that relies on, assumes, or falls back to home charging as unrealistic.

From the person's demographics, trips, and charging access, reason through what a real person like this would most plausibly do. Penalize choices inconsistent with their schedule, household, and budget, as well as ignoring charger availability/contention, level-vs-time mismatches (expecting a full charge in a short stop), economically irrational choices, and any reliance on home charging.

Give your reasoning and the ranking as a list of person numbers from most to least realistic (e.g. [2, 1, 3, 5, 4]). Include every person exactly once."""
    user_prompt = f"""Profile:
{profile_str}

Scenario:
{scenario}

{responses_text}
Rank the people."""

    result = client.responses.parse(
        model = "gpt-5.4-mini",
        reasoning = {"effort": "high"},
        input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text_format = RankingResult
    )
    print(result.output_parsed, end = "\n\n\n")
    return result.output_parsed

def generate_best_persona(target: Profile, client: OpenAI) -> PersonaArtifact:
    dispositions = DISPOSITIONS[target.archetype]
    scenarios = SCENARIOS[target.archetype]
    personas = [generate_persona(target, client, disposition) for disposition in dispositions]
    n_samples = len(personas)
    actions = [[answer_scenario(persona, scenario, client) for scenario in scenarios] for persona in personas]

    scores = [0] * n_samples
    rankings: list[ScenarioRanking] = []

    for index in range(len(scenarios)):
        responses = [actions[persona_index][index] for persona_index in range(n_samples)]
        result = rank_personas(target, scenarios[index], responses, client)
        order = [label - 1 for label in result.ranking]
        if sorted(order) != list(range(n_samples)):
            raise ValueError(f"Ranking {result.ranking} is not a permutation of 1..{n_samples}")
        for position, persona_index in enumerate(order):
            scores[persona_index] += n_samples - 1 - position
        rankings.append(ScenarioRanking(
            scenario = scenarios[index],
            reasoning = result.reasoning,
            ranking = order
        ))

    candidates = [
        PersonaCandidate(
            persona = personas[persona_index],
            disposition = dispositions[persona_index],
            actions = actions[persona_index],
            score = scores[persona_index]
        )
        for persona_index in range(n_samples)
    ]
    best_index = max(range(n_samples), key = lambda persona_index: scores[persona_index])
    print(f"Best persona {best_index}: score={candidates[best_index].score}")

    return PersonaArtifact(
        personas = candidates,
        best_index = best_index,
        target_profile = target,
        rankings = rankings
    )

if __name__ == "__main__":
    dotenv.load_dotenv(override = True)
    client = OpenAI()

    with open("artifacts/profiles.json") as file:
        profiles_json = json.load(file)

    profiles: list[Profile] = []

    for profile_json in profiles_json:
 
        profile = Profile.model_validate(profile_json)
        if all(trip.vehicle is not None and trip.vehicle.fuel_type in {"Electric", "Plug-in Hybrid"} for trip in profile.trips):
            profiles.append(profile)
    
    artifacts = []
    for archetype in Archetype:
        target = [profile for profile in profiles if profile.archetype == archetype][0]
        artifact = generate_best_persona(target, client)
        best = artifact.personas[artifact.best_index]
        print(f"{archetype.value} best persona {artifact.best_index}: score={best.score}")
        artifacts.append(json.loads(artifact.model_dump_json()))

    with open("artifacts/personas.json", "w") as file:
        json.dump(artifacts, file, indent = 2)

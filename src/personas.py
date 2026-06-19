import dotenv
import json
from openai import OpenAI
from pydantic import BaseModel
from profiles import Profile, Archetype

class ScenarioResponse(BaseModel):
    reasoning: str
    action: str

class RankingResult(BaseModel):
    reasoning: str
    ranking: list[int]

class ScenarioRanking(BaseModel):
    scenario: str
    reasoning: str
    ranking: list[int]

class PersonaCandidate(BaseModel):
    persona: str
    actions: list[ScenarioResponse]
    score: int

class PersonaArtifact(BaseModel):
    personas: list[PersonaCandidate]
    best_index: int
    target_profile: Profile
    charging_rankings: list[ScenarioRanking]

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

def generate_persona(profile: Profile, client: OpenAI) -> str:
    system_prompt = """You are a social scientist building grounded, realistic personas of real people for a travel-behavior study. You will be given a Profile with demographics and a list of trips taken on a single day. Your persona must plausibly fit the Profile and make the person's routine, day-to-day variation, and EV charging habits inferable.

DON'T:
- Restate the trip list, time-stamped itineraries, or hour-by-hour schedules.
- Restate age group, household income, employment status, student status, or household size verbatim.
- Lock the person into one fixed daily schedule — the trip list is one day out of many.
- Assume home charging: this person has NO home charging and charges only away from home.
- State exact kWh, exact battery percentages at exact times, or a fixed charging schedule.
- Hedge. Do not use "probably", "likely", or "suggests"; state tendencies as fact.

DO:
- Give general demographics that help infer the person's trip schedule.
- Give time windows (early morning, late afternoon) and usual activities, each with a real-world reason.
- Separate firm anchors (work start, school pickup) from loose discretionary activities (gym, errands, meals).
- Give frequencies, tendencies, and conditional day-to-day variation, not single occurrences.
- Give preferences and trade-offs so the person's choices can be inferred in new situations.
- Convey their away-from-home EV charging: where they plug in (a dedicated stop vs topping up while already stopped), how low they let the battery get, fast DC vs slower top-up, how they react to a busy charger (wait, relocate, or skip), and price sensitivity.

EXAMPLES:

Bad: He leaves the house at 6:30 am
Good: He gets up very early in the morning to avoid traffic

Bad: She has a pick-up trip at 3:30 pm
Good: She picks up her kids from school in the mid-afternoon, around when school typically ends

Bad: He takes a break from work and goes to a fast food restaurant around 12:30 pm
Good: He does not like cooking at home and values convenience, especially for lunch

Bad: Some weeks he skips the gym on Tuesday
Good: Some weeks he skips the gym when work runs long

Bad: He charges to 80% at 2:15 pm
Good: He tops up while already stopped for an errand and avoids letting the battery get too low

Bad: She charges every day right after work
Good: On long-driving days she seeks out a fast charger; most days she charges while running errands
"""

    profile_str = format_profile(profile)
    user_prompt = f"""Profile:
{profile_str}

Generate one persona from this profile."""

    response = client.responses.create(
        model = "gpt-5.4-mini",
        input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    print(response.output_text, end = "\n\n\n")
    return response.output_text

CHARGING_SCENARIOS = [
    "All the chargers at the stop where you planned to charge are busy when you arrive. Do you wait for one to free up, drive to a different stop to charge, or skip charging today?",
    "You only have a short stop before you need to move on, but your battery is low. Do you pay for a fast DC charge now, or do a slower charge while parked somewhere later in the day?",
    "You will pass several stops today where you could plug in. Which one do you choose to charge at, and why?",
    "Public charging prices have spiked today. Does that change where, when, or whether you charge?",
    "Your battery is low and you have a long stretch of driving ahead with no stop already planned. What do you do?"
]

def answer_scenario(persona: str, scenario: str, client: OpenAI) -> ScenarioResponse:
    response = client.responses.parse(
        model = "gpt-5.4-mini",
        input = [
            {"role": "system", "content": f"""You are the following persona. Stay fully in character as this person.

Persona:
{persona}

You have no way to charge at home, so all of your charging happens away from home during the day.

You will be given a situation. Respond with the charging action you actually take and why, true to who you are. Do not narrate as a third party."""},
            {"role": "user", "content": scenario}
        ],
        text_format = ScenarioResponse
    )
    print(response.output_parsed, end = "\n\n\n")
    return response.output_parsed

def rank_personas(profile: Profile, scenario: str, responses: list[ScenarioResponse], client: OpenAI) -> RankingResult:
    profile_str = format_profile(profile)
    responses_text = "\n".join(
        f"Person {index + 1} action:\n{response.action}\n\nPerson {index + 1} reasoning:\n{response.reasoning}\n"
        for index, response in enumerate(responses)
    )
    result = client.responses.parse(
        model = "gpt-5.4-mini",
        reasoning = {"effort": "high"},
        input = [
            {"role": "system", "content": """You are an EV charging behavior expert. Several people each describe how they would handle the same charging situation. Rank them from most to least realistic for the actual person described in the profile.

Assume NO home charging exists in this study — nobody can charge at home. Treat any action or reasoning that relies on, assumes, or falls back to home charging as unrealistic.

From the person's demographics, trips, and charging access (workplace EV charging when stated), reason through what a real person like this would most plausibly do away from home. Penalize ignoring charger availability/contention, level-vs-time mismatches (expecting a full charge in a short stop), economically irrational choices, and any reliance on home charging.

Give your reasoning and the ranking as a list of person numbers from most to least realistic (e.g. [2, 1, 3]). Include every person exactly once."""},
            {"role": "user", "content": f"""Profile:
{profile_str}

Scenario:
{scenario}

{responses_text}
Rank the people."""}
        ],
        text_format = RankingResult
    )
    print(result.output_parsed, end = "\n\n\n")
    return result.output_parsed

def generate_best_persona(target: Profile, client: OpenAI, n_samples: int = 3) -> PersonaArtifact:
    personas = [generate_persona(target, client) for _ in range(n_samples)]
    actions = [[answer_scenario(persona, scenario, client) for scenario in CHARGING_SCENARIOS] for persona in personas]

    scores = [0] * n_samples
    rankings: list[ScenarioRanking] = []

    for index in range(len(CHARGING_SCENARIOS)):
        responses = [actions[persona_index][index] for persona_index in range(n_samples)]
        result = rank_personas(target, CHARGING_SCENARIOS[index], responses, client)
        order = [label - 1 for label in result.ranking]
        if sorted(order) != list(range(n_samples)):
            raise ValueError(f"Ranking {result.ranking} is not a permutation of 1..{n_samples}")
        for position, persona_index in enumerate(order):
            scores[persona_index] += n_samples - 1 - position
        rankings.append(ScenarioRanking(
            scenario = CHARGING_SCENARIOS[index],
            reasoning = result.reasoning,
            ranking = order
        ))

    candidates = [
        PersonaCandidate(
            persona = personas[persona_index],
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
        charging_rankings = rankings
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
    
    target = [profile for profile in profiles if profile.archetype == Archetype.FLEXIBLE_COMMUTER][0]
    artifact = generate_best_persona(target, client)
    best = artifact.personas[artifact.best_index]
    print(best.persona)
    print(f"Charging score: {best.score}")

    with open("artifacts/personas.json", "w") as file:
        json.dump([json.loads(artifact.model_dump_json())], file, indent = 2)

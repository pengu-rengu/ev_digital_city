import dotenv
import json
import random
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel
from profiles import Profile, Archetype

class PersonaChoice(BaseModel):
    reasoning_first: str
    reasoning_second: str
    choice: Literal["A", "B"]

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
    
    text += "\nTrips:\n\n"

    for trip in profile.trips:
        text += f"Departure Time: {trip.departure_time}\n"
        text += f"Arrival time: {trip.arrival_time}\n"
        text += f"Origin Activity: {trip.origin_activity}\n"
        text += f"Destination Activity: {trip.dest_activity}\n"
        text += f"Travel Distance: {trip.distance} miles\n"
        text += f"Travel Time: {trip.travel_time} minutes\n\n"

    return text

def generate_persona(profile: Profile, peers: list[Profile], client: OpenAI, best_persona: str | None, worst_persona: str | None) -> str:
    peers_block = "\n\n".join(
        "Peer Profile {index}:\n{profile_text}".format(index = i + 1, profile_text = format_profile(peer))
        for i, peer in enumerate(peers)
    )

    system_prompt = """You are a social scientist building grounded, realistic personas of real people for a travel-behavior study.

You will be given a Profile, which contains demographics and a list of trips taken on a single travel-diary day.
You will also be given Peer Profiles: other people whose demographics look similar to the target. Your persona must plausibly fit the target Profile and clearly NOT fit the Peer Profiles. Lean on whatever distinguishes the target's trips and demographics from the peers.

Do not exactly restate the trip list. Do not produce time-stamped itineraries or hour-by-hour schedules. 
Do not exactly restate age group, household income, employment status, student status, or household size.
This information is evidence about the person, not the persona itself.

Be confident in your claims. Do not use words like "probably", "likely", or "suggests"

Your persona should structured as follows:

Occupation and life stage: 2 sentences
Household and social role: 2 sentences
Values and Motivations: 3 sentences
Typical activities and schedule: 3 sentences

Peer Profiles to differentiate from:
{peers_block}
""".format(peers_block = peers_block)

    if best_persona is not None:
        system_prompt += """
The best previous attempt is below. It fit the target better than peers but was still too generic. Improve on it, keep what differentiates, sharpen the rest.

Best previous persona:
{best_persona}
""".format(best_persona = best_persona)

    if worst_persona is not None:
        system_prompt += """
The worst previous attempt is below. It was the least distinguishing, it fit the peers about as well as the target. Avoid its framing and generic claims.

Worst previous persona:
{worst_persona}
""".format(worst_persona = worst_persona)
    
    print(system_prompt)

    response = client.responses.create(
        model = "gpt-5.4-mini",
        reasoning = {
            "effort": "medium",
            "summary": "auto"
        },
        input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": """Profile:
{profile}

Generate one persona.""".format(profile = format_profile(profile))}
        ]
        #service_tier = "flex"
    )
    reasoning = next(item for item in response.output if item.type == "reasoning")
    print("\n\n".join([summary.text for summary in reasoning.summary]))
    print(response.output_text)
    return response.output_text

def score_persona(anchor: Profile, peer: Profile, persona: str, client: OpenAI) -> int:
    anchor_is_a = random.random() < 0.5
    profile_a = anchor if anchor_is_a else peer
    profile_b = peer if anchor_is_a else anchor
    first_letter = "A" if random.random() < 0.5 else "B"
    second_letter = "B" if first_letter == "A" else "A"

    response = client.responses.parse(
        model = "gpt-5.4-mini",
        input = [
            {"role": "system", "content": """You are the following persona. Stay fully in character as this person.

Persona:
{persona}

You will be shown two travel-diary Profiles labelled A and B: each contains demographics and a list of trips taken on a single day. Only one of them is yours.

First explain why Profile {first_letter} could fit you in `reasoning_first`, then explain why Profile {second_letter} could fit you in `reasoning_second` (1-2 sentences each). Then pick the letter (A or B) that is more likely to be yours.""".format(persona = persona, first_letter = first_letter, second_letter = second_letter)},
            {"role": "user", "content": """Profile A:
{profile_a}

Profile B:
{profile_b}

Which Profile is more likely yours?""".format(profile_a = format_profile(profile_a), profile_b = format_profile(profile_b))}
        ],
        text_format = PersonaChoice
    )
    choice = response.output_parsed
    print(choice)
    anchor_chosen = (choice.choice == "A") == anchor_is_a
    return 1 if anchor_chosen else 0

def profile_similarity(profile_a: Profile, profile_b: Profile) -> int:
    excluded = {"archetype", "attributes", "trips"}
    return sum(
        1 for field in Profile.model_fields
        if field not in excluded and getattr(profile_a, field) == getattr(profile_b, field)
    )

def refine_persona(profile: Profile, profiles: list[Profile], threshold: float, client: OpenAI, max_iterations: int = 5) -> tuple[str, float]:
    peers_pool = [other for other in profiles if other.archetype == profile.archetype and other is not profile]
    others = sorted(peers_pool, key = lambda other: profile_similarity(profile, other), reverse = True)[:3]
    anchor = profile

    best_persona: str | None = None
    best_score = float("-inf")
    worst_persona: str | None = None
    worst_score = float("inf")

    for iteration in range(max_iterations):
        persona = generate_persona(anchor, others, client, best_persona, worst_persona)
        wins = sum(score_persona(anchor, other, persona, client) for other in others)
        final_score = wins / len(others)
        print(f"Iteration {iteration}: wins={wins}/{len(others)} final={final_score:.2f}")

        if final_score > best_score:
            best_score = final_score
            best_persona = persona

        if final_score < worst_score:
            worst_score = final_score
            worst_persona = persona

        if final_score >= threshold:
            return persona, final_score

    return best_persona, best_score

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
    
    anchor = [profile for profile in profiles if profile.archetype == Archetype.RIGID_COMMUTER][3]
    persona, final_score = refine_persona(
        anchor,
        profiles,
        threshold = 1.0,
        client = client
    )
    print(persona)
    print(f"Final score: {final_score:.2f}")

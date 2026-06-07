import dotenv
from openai import OpenAI
from pydantic import BaseModel
from profiles import Profile, Archetype


class PersonaScore(BaseModel):
    fits: str
    does_not_fit: str
    score: float

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

def generate_persona(profile: Profile, client: OpenAI) -> str:
    response = client.responses.create(
        model = "gpt-5.4-mini",
        input = [
            {"role": "system", "content": """You are a social scientist building grounded, realistic personas of real people for a travel-behavior study.

You will be given a Profile, which contains demographics and a list of trips taken on a single travel-diary day.
Your task is to generate a persona of a person who could plausibly be behind this Profile.
Do not restate the trip list. Do not produce time-stamped itineraries or hour-by-hour schedules. The trips are evidence about the person, not the persona itself.

Your persona should structured as follows:

Occupation and life stage: 2 sentences
Household and social role: 2 sentences
Values and Motivations: 3 sentences
Typical activities and schedule: 3 sentences
"""},
            {"role": "user", "content": """Profile:
{profile}

Generate one persona.""".format(profile = format_profile(profile))}
        ]
        #service_tier = "flex"
    )
    return response.output_text

def score_persona(profile: Profile, persona: str, client: OpenAI) -> PersonaScore:
    response = client.responses.parse(
        model = "gpt-5.4-mini",
        input = [
            {"role": "system", "content": """You are the following persona. Stay fully in character as this person.

Persona:
{persona}

You will be shown a travel-diary Profile: demographics and a list of trips taken on a single day.
Rate, on a 1.0-10.0 scale, how likely it is that this Profile describes you.

10.0 = this Profile is almost certainly mine; demographics and trips fit me closely.
1.0 = this Profile clearly does not describe me.

First explain why this Profile fits you, then explain why it does not fit you (1-2 sentences each). Then give the score.""".format(persona = persona)},
            {"role": "user", "content": """Profile:
{profile}

Rate how likely this Profile matches you.""".format(profile = format_profile(profile))}
        ],
        text_format = PersonaScore
    )
    return response.output_parsed


if __name__ == "__main__":
    import json

    dotenv.load_dotenv(override = True)
    client = OpenAI()

    with open("artifacts/profiles.json") as file:
        profiles_json = json.load(file)

    profiles: list[Profile] = []

    for profile_json in profiles_json:
 
        profile = Profile.model_validate(profile_json)
        if all(trip.vehicle is not None and trip.vehicle.fuel_type in {"Electric", "Plug-in Hybrid"} for trip in profile.trips):
            profiles.append(profile)
        

    for profile in profiles:
        if profile.archetype != Archetype.FLEXIBLE_COMMUTER:
            continue
        print(format_profile(profile))
        print(f"Archetype: {profile.archetype.value}")
        print(f"Age group: {profile.age_group}, employment: {profile.employment_status}")
        print()

        persona = generate_persona(profile, client)
        print(persona)
        score = score_persona(profile, persona, client)
        
        print(score)
        break

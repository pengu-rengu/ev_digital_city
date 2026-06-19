import dotenv
import json
from openai import OpenAI
from pydantic import BaseModel
from profiles import Profile, Archetype

class PersonaArtifact(BaseModel):
    persona: str
    target_profile: Profile

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
    system_prompt = """You are a social scientist building grounded, realistic personas of real people for a travel-behavior study.

You will be given a Profile, which contains demographics and a list of trips taken on a single travel-diary day.
Your persona must plausibly fit the target Profile. Make the person's routine and the rhythm of their day inferable from the persona.

DON'T:
Do not exactly restate the trip list. Do not produce time-stamped itineraries or hour-by-hour schedules.
Do not exactly restate age group, household income, employment status, student status, or household size.

DO:
Do provide general demographics that would help infer this person's trip schedule
Do provide time windows (e.g. early morning, late afternoon), along with a real world explanation
Do provide activities the person usually does, along with a real world explanation

ADD VARIATION:
The trip list is ONE day out of many. Describe the person's typical pattern and how it shifts day to day, not a single fixed schedule.
- Separate non-negotiable anchors (work start, school pickup) from discretionary activities (gym, errands, meals). Make anchors clear and firm; leave discretionary activities loose in timing and presence.
- Convey what is routine versus what changes. Example: "Some weeks she skips the gym.")
- Give frequencies and tendencies, not single occurrences. ("Shops two or three times a week", not one shopping trip.)
- Give the person's preferences and trade-offs (values convenience, avoids rush hour, prefers charging while already stopped) so their choices can be inferred in different situations, rather than stating fixed outcomes.
- Describe conditional behavior. ("On busy days he eats out; otherwise he cooks at home.")
- Convey the rough load of the day (a few errands around work), not an exact enumerated chain of stops in fixed order.

EXAMPLES:

Bad: He leaves the house at 6:30 am
Good: He gets up very early in the morning to avoid traffic

Bad: She has a pick-up trip at 3:30 pm
Good: She pick ups her kids from school in the mid-afternoon, around when school typically ends

Bad: He takes a break from work and goes to a fast food restaurant around 12:30 pm
Good: He does not like cooking at home and values convenience, especially for lunch

This information is evidence about the person, not the persona itself.
Be confident about the person's tendencies, including what varies day to day. State variation as fact, not as hedging about your guess. Do not use words like "probably", "likely", or "suggests".
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
    persona = generate_persona(target, client)
    artifact = PersonaArtifact(persona = persona, target_profile = target)
    print(artifact.persona)

    with open("artifacts/personas.json", "w") as file:
        json.dump([json.loads(artifact.model_dump_json())], file, indent = 2)

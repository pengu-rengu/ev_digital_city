import dotenv
import json
from openai import OpenAI
from pydantic import BaseModel
from profiles import Profile, Archetype, arrival_minutes

class TripGuess(BaseModel):
    origin_activity: str
    dest_activity: str
    departure_time: str
    arrival_time: str

class TrajectoryGuess(BaseModel):
    reasoning: str
    trips: list[TripGuess]

class Iteration(BaseModel):
    persona: str
    guess: TrajectoryGuess
    final_score: float

class PersonaArtifact(BaseModel):
    best_persona: str
    target_profile: Profile
    iterations: list[Iteration]

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

def generate_persona(profile: Profile, client: OpenAI, last_persona: str | None, reflection: str | None) -> str:
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

EXAMPLES:

Bad: He leaves the house at 6:30 am
Good: He gets up very early in the morning to avoid traffic

Bad: She has a pick-up trip at 3:30 pm
Good: She pick ups her kids from school in the mid-afternoon, around when school typically ends

Bad: He takes a break from work and goes to a fast food restaurant around 12:30 pm
Good: He does not like cooking at home and values convenience, especially for lunch

This information is evidence about the person, not the persona itself.
Be confident in your claims. Do not use words like "probably", "likely", or "suggests".
"""

    if last_persona is not None:
        system_prompt += """
Your previous persona attempt is below, along with guidance on how to improve it. Produce a better persona that applies the guidance, keeping what worked and fixing what the guidance points out.

Previous persona:
{last_persona}

Guidance:
{reflection}
""".format(last_persona = last_persona, reflection = reflection)

    response = client.responses.create(
        model = "gpt-5.4-mini",
        input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": """Profile:
{profile}

Generate one persona.""".format(profile = format_profile(profile))}
        ]
        #service_tier = "flex"
    )
    print(response.output_text, end = "\n\n\n")
    return response.output_text

def within_tolerance(guess: str, actual: str, tolerance: int = 30) -> bool:
    return abs(arrival_minutes(guess) - arrival_minutes(actual)) <= tolerance

def score_persona(target: Profile, persona: str, client: OpenAI) -> tuple[TrajectoryGuess, float]:
    num_trips = len(target.trips)
    response = client.responses.parse(
        model = "gpt-5.4-mini",
        input = [
            {"role": "system", "content": """You are the following persona. Stay fully in character as this person.

Persona:
{persona}

You took {num_trips} trips on your travel-diary day. Reconstruct them in order. For each trip, give the origin activity, destination activity, departure time (HH:MM), and arrival time (HH:MM).

Each activity must be one of: Home, Work, Volunteer, School, Shopping, Meal (quick-stop), Meal, Gas, Health care, Non-shopping errand, Socialize, Civic/Religious, Exercise, Recreation, Entertainment, Drop off/pick up, Other.""".format(persona = persona, num_trips = num_trips)},
            {"role": "user", "content": "Reconstruct your {num_trips} trips.".format(num_trips = num_trips)}
        ],
        text_format = TrajectoryGuess
    )
    print(response.output_parsed, end = "\n\n\n")
    guess = response.output_parsed

    hits = 0
    for guessed, actual in zip(guess.trips, target.trips):
        hits += guessed.origin_activity == actual.origin_activity
        hits += guessed.dest_activity == actual.dest_activity
        hits += within_tolerance(guessed.departure_time, actual.departure_time)
        hits += within_tolerance(guessed.arrival_time, actual.arrival_time)

    final_score = hits / (4 * num_trips)
    return guess, final_score

def reflect(target: Profile, persona: str, guess: TrajectoryGuess, client: OpenAI) -> str:
    guess_text = "\n\n".join(
        "Trip {index}:\nDeparture Time: {departure}\nArrival Time: {arrival}\nOrigin Activity: {origin}\nDestination Activity: {dest}".format(
            index = i + 1,
            departure = trip.departure_time,
            arrival = trip.arrival_time,
            origin = trip.origin_activity,
            dest = trip.dest_activity
        )
        for i, trip in enumerate(guess.trips)
    )

    response = client.responses.create(
        model = "gpt-5.4-mini",
        reasoning = {
            "effort": "medium",
            "summary": "auto"
        },
        input = [
            {"role": "system", "content": """You are improving a persona used in a travel-behavior study. Someone role-playing the persona tried to reconstruct the person's real trips (which activities they went to and roughly when) and produced a guess. You are given the persona, that guess, and the person's actual trips.

Explain concretely how the persona should be revised so a reader reconstructs the trips more accurately: which activities and timing the persona fails to convey, and what it should make inferable instead.

Do not restate or quote the actual profile or trip values. Give guidance about what the persona should convey, not the answers themselves."""},
            {"role": "user", "content": """Persona:
{persona}

Reconstruction guess:
{guess_text}

Actual:
{actual}

How should the persona be improved?""".format(persona = persona, guess_text = guess_text, actual = format_profile(target))}
        ]
    )
    print(response.output_text, end="\n\n\n")
    return response.output_text

def refine_persona(target: Profile, threshold: float, client: OpenAI, max_iterations: int = 8) -> tuple[str, float]:
    best_persona: str | None = None
    best_score = float("-inf")
    last_persona: str | None = None
    reflection: str | None = None

    for iteration in range(max_iterations):
        persona = generate_persona(target, client, last_persona, reflection)
        guess, final_score = score_persona(target, persona, client)
        print(f"Iteration {iteration}: final={final_score:.2f}")

        if final_score > best_score:
            best_score = final_score
            best_persona = persona

        if final_score >= threshold:
            return persona, final_score

        if iteration < max_iterations - 1:
            reflection = reflect(target, persona, guess, client)
            last_persona = persona

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
    
    target = [profile for profile in profiles if profile.archetype == Archetype.FLEXIBLE_COMMUTER][3]
    persona, final_score = refine_persona(
        target,
        threshold = 1.0,
        client = client
    )
    print(persona)
    print(f"Final score: {final_score:.2f}")

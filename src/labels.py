INCOME_LABELS = {
    1: "Less than $15,000",
    2: "$15,000 - $24,999",
    3: "$25,000 - 34,999",
    4: "$35,000 - 49,999",
    5: "$50,000 - $74,999",
    6: "$75,000 - $99,999",
    7: "$100,000 - $149,999",
    8: "$150,000 or more",
}

AGE_GROUP_LABELS = {
    1: "Under 5 years old",
    2: "5 - 11 years",
    3: "12 - 13 years",
    4: "14 - 15 years",
    5: "16 - 17 years",
    6: "18 - 24 years",
    7: "25 - 34 years",
    8: "35 - 44 years",
    9: "45 - 54 years",
    10: "55 - 64 years",
    11: "65 - 74 years",
    12: "75 - 84 years",
    13: "85 years or older",
}

EMPLOYMENT_LABELS = {
    0: "Worker, including self employed",
    1: "Retired",
    2: "Volunteer",
    3: "Homemaker",
    4: "Unemployed but looking for work",
    5: "Unemployed, not seeking employment",
    6: "Student (part-time or full-time)",
    7: "Disabled non-worker",
    -9: None,
}

STUDENT_STATUS_LABELS = {
    0: "Not a student",
    1: "Student",
    -9: None,
}

WORKPLACE_LOC_LABELS = {
    1: "Usually the same location (outside home)",
    2: "Workplace regularly varies (different offices or jobsites)",
    3: "At home (telecommute or self-employed with home office)",
    4: "Drives for a living (e.g., bus driver, salesperson)",
    -9: None,
}

TELECOMMUTE_DAYS_LABELS = {
    0: "0 days",
    1: "1 day",
    2: "2 days",
    3: "3 days",
    4: "4 days",
    5: "5+ days",
    -9: None,
}

COMMUTE_FREQ_LABELS = {
    1: "5 weekdays a week",
    2: "4 weekdays a week",
    3: "3 weekdays a week",
    4: "2 weekdays a week",
    5: "1 weekday a week",
    6: "Weekends only",
    7: "Less than four weekdays per month",
    8: "Schedule varies week-to-week",
    -9: None,
}

SCHOOL_FREQ_LABELS = {
    1: "5 weekdays a week",
    2: "4 weekdays a week",
    3: "3 weekdays a week",
    4: "2 weekdays a week",
    5: "1 weekday a week",
    6: "Weekends only",
    7: "Less than four weekdays per month",
    8: "Never, only takes online classes (if age 5+)",
    -9: None,
}

SCHOOL_TYPE_LABELS = {
    1: "Daycare / Nursery / Pre-school",
    2: "K - 8th grade",
    3: "9th - 12th grade",
    4: "Home school",
    5: "Vocational/Technical school",
    6: "2-year college (community college)",
    7: "4-year college or university",
    8: "Graduate/Professional school",
    9: "Other",
    -9: None,
}

HOME_TYPE_LABELS = {
    1: "Single-family house (detached)",
    2: "Single-family house (attached)",
    3: "Apartment/Condo",
    4: "Mobile home/trailer",
    5: "Dorm or institutional housing",
    -9: None,
}

HOME_OWNERSHIP_LABELS = {
    1: "Own",
    2: "Rent",
    3: "Other",
    -9: None,
}

GENDER_LABELS = {
    1: "Female",
    2: "Male",
    -9: None,
}

RACEETHNICITY_LABELS = {
    1: "Hispanic or Latino",
    2: "African American or Black",
    3: "Asian",
    4: "White",
    5: "Other/Two or more races",
    -9: None,
}

TRAVEL_MODE_LABELS = {
    1: "Walk",
    2: "Bike",
    3: "Motorcycle",
    4: "Auto (driver)",
    5: "Auto (passenger)",
    6: "School Bus",
    7: "Rail",
    8: "Bus",
    9: "Private Bus",
    10: "Paratransit",
    11: "Taxi / Private Car",
    12: "Uber/Lyft/Rideshare",
    13: "Air",
    14: "Water",
    15: "Other",
}

FUELTYPE_LABELS = {
    1: "Gas",
    2: "Diesel",
    3: "Plug-in Hybrid",
    4: "Hybrid",
    5: "Electric",
    6: "Flex Fuel",
    997: "Other",
    -9: None,
}

BODYTYPE_LABELS = {
    1: "Car (or station wagon)",
    2: "Van (any type)",
    3: "SUV",
    4: "Pickup Truck",
    5: "Other type of truck",
    6: "RV",
    7: "Motorcycle",
    997: "Other",
    -9: None,
}

ACTIVITY_LABELS = {
    1: "Home",
    2: "Work",
    3: "Volunteer",
    4: "School",
    5: "Shopping",
    6: "Meal (quick-stop)",
    7: "Meal",
    8: "Gas",
    9: "Health care",
    10: "Non-shopping errand",
    11: "Socialize",
    12: "Civic/Religious",
    13: "Exercise",
    14: "Recreation",
    15: "Entertainment",
    16: "Drop off/pick up",
    18: "Other",
    -9: None,
}
from profiles import Archetype

DISPOSITIONS: dict[Archetype, list[str]] = {
    Archetype.NON_COMMUTER: [
        "Thrifty patient retiree. Lets the battery run low, waits for cheap L2 while parked at an appointment or shopping center, never pays a DC premium, and happily trades time for savings.",
        "Convenience-first affluent retiree. Values ease over cost, pays DC to get home sooner, and leaves a busy station immediately rather than wait.",
        "Range-averse cautious retiree. Tops up early and well above empty, plans charging around predictable stops, and is anxious about longer medical drives.",
        "Spontaneous social retiree. Day shifts with family visits and social plans, adds unplanned trips, and charges opportunistically wherever already stopped."
    ],
    Archetype.PARENT_COMMUTER: [
        "Childcare-first juggler. Drops everything for pickups, chains errands tightly around school and activity runs, and abandons charging the instant a kid obligation appears.",
        "Shared-vehicle coordinator. Must finish charging before the household handoff, plans around the partner's schedule, and picks DC when the car is needed back soon.",
        "Budget-stretched family driver. Price-sensitive, defers to cheap charging, bundles family errands to save trips and money, and tolerates a slow charge during a long kid activity.",
        "Time-pressed convenience parent. Pays the DC premium to hold the tight school-and-work schedule, and won't wait at a busy charger with kids in the car."
    ],
    Archetype.FLEXIBLE_COMMUTER: [
        "Off-peak optimizer. Shifts trips and charging to cheap off-peak windows on remote days, and plans around price and grid signals.",
        "Spontaneous remote-day errand-runner. Loose time-budget, adds stops on a whim, and charges mid-errand wherever convenient.",
        "Office-day convenience charger. Only charges on heavy in-office days, pays DC to keep the commute tight, and relocates if a station is busy.",
        "Frugal homebody hybrid. Minimizes trips on remote days, lets the battery sit, hunts cheap L2 when out, and rarely fast-charges."
    ],
    Archetype.RIGID_COMMUTER: [
        "Schedule-protecting fast-charger. Abandons busy chargers, pays DC for predictability, and never detours off the fixed route.",
        "Routine cost-minimizer. Charges at the same cheap, predictable spot on a fixed cadence; won't pay a premium and won't improvise.",
        "Range-anxious commuter. Tops up well before empty to never risk the commute, and keeps a standing charging stop.",
        "Corridor-dependent charger. Leans on one reliable charger on the commute corridor, has low tolerance for a queue, and is stressed when it is down."
    ]
}

SCENARIOS: dict[Archetype, list[str]] = {
    Archetype.NON_COMMUTER: [
        "A mid-morning medical visit, your battery is low, and the charger lot is full. (A) Wait for a stall since you'll be parked for the appointment anyway, (B) skip and use a cheaper charger near your afternoon shopping, or (C) pay for a fast DC charge nearby to be safe before the drive home.",
        "A free L2 charger is 10 minutes farther; a paid DC charger sits right on your route. (A) Detour for the free L2 while you shop, (B) take the quick paid DC on route, or (C) skip charging since your local range is fine.",
        "Your battery hits 80% and charging slows, while a long family-visit drive looms tomorrow. (A) Unplug at 80% now, (B) wait for 100% to cover tomorrow, or (C) leave at 80% and fast-charge en route tomorrow."
    ],
    Archetype.PARENT_COMMUTER: [
        "School calls for an early pickup during your planned charging stop, and your battery is low. (A) Abandon the charge and go straight to school, (B) take a quick DC top-up first if it's on the way, or (C) ask your partner to cover while you finish a short charge.",
        "Your partner needs the shared car by 6 PM; you still have errands and a low battery. (A) Fast DC now to hand it back full, (B) cheap slow charge and drop an errand, or (C) skip charging, finish errands, and hand it back low.",
        "Three kid runs plus groceries today. (A) Chain everything into one loop and charge during the longest activity, (B) make separate trips as each comes up and charge opportunistically, or (C) one loop but DC so you're never caught low with kids in the car."
    ],
    Archetype.FLEXIBLE_COMMUTER: [
        "It's a remote day and public charging prices spike midday. (A) Shift charging to the cheap off-peak evening, (B) charge now to keep the rest of the day free, or (C) skip and charge tomorrow on the office run.",
        "A friend proposes a spontaneous cross-town dinner after work. (A) Go, and fast-charge en route first, (B) go only if your current range covers it, or (C) decline since your range is too tight.",
        "Your errands could be bundled today or deferred to a remote day. (A) Bundle now and charge at the longest stop, (B) defer to a low-traffic remote day, or (C) do them now and pay DC to stay flexible."
    ],
    Archetype.RIGID_COMMUTER: [
        "You have a fixed start time, the charger on your route is full, and the queue is uncertain. (A) Wait and risk being late, (B) abandon it and drive to another charger on route, or (C) skip and charge after work at your usual spot.",
        "A cheaper charger is 10 minutes off your fixed route; your usual one is on-route but pricier. (A) Stick to the on-route pricier one, (B) detour for the cheaper one, or (C) alternate depending on how tight the morning is.",
        "An unexpected expense hit your household this month and charging is a notable cost. (A) Switch to the cheaper off-route charger despite the detour, (B) keep the routine and trim other spending, or (C) cut non-commute driving but keep the commute charge."
    ]
}

"""
SOURCES

Dispositions and scenarios above are grounded in the deep-research reports in
artifacts/ (report_a/b/c.md). Each item is labeled with the citation number(s)
it draws on; full citations follow.

NON_COMMUTER dispositions:
  Thrifty patient retiree ............ [1][2]
  Convenience-first affluent retiree . [3][4]
  Range-averse cautious retiree ...... [2]
  Spontaneous social retiree ......... [7][8]
NON_COMMUTER scenarios:
  1 medical visit, charger lot full .. [1][15]
  2 free L2 vs paid DC on route ...... [3][5]
  3 80% taper before long drive ...... [6]

PARENT_COMMUTER dispositions:
  Childcare-first juggler ............ [9][11]
  Shared-vehicle coordinator ......... [10][11]
  Budget-stretched family driver ..... [5][12]
  Time-pressed convenience parent .... [2][4]
PARENT_COMMUTER scenarios:
  1 early pickup vs charge ........... [9][11]
  2 partner needs car by 6 PM ........ [10]
  3 three kid runs + groceries ....... [12][13]

FLEXIBLE_COMMUTER dispositions:
  Off-peak optimizer ................. [4][16]
  Spontaneous remote-day runner ...... [7][13]
  Office-day convenience charger ..... [4][16]
  Frugal homebody hybrid ............. [2][5]
FLEXIBLE_COMMUTER scenarios:
  1 price spike on a remote day ...... [4]
  2 spontaneous cross-town dinner .... [7][8]
  3 bundle today vs defer ............ [12][13]

RIGID_COMMUTER dispositions:
  Schedule-protecting fast-charger ... [13][14]
  Routine cost-minimizer ............. [3][5]
  Range-anxious commuter ............. [2]
  Corridor-dependent charger ......... [15]
RIGID_COMMUTER scenarios:
  1 route charger full, fixed start .. [14][15]
  2 cheaper off-route vs on-route .... [3]
  3 unexpected expense ............... [2][5]

CITATIONS

[1] Hardman, S. (2026). "Exploring Electric Vehicle Driver Activities and Expenditure While using DC Fast Chargers." Findings. DOI: 10.32866/001c.118635.
[2] Liu, Y. S., Tayarani, M., & Gao, H. O. (2022). "An Activity-Based Travel and Charging Behavior Model for Simulating Battery Electric Vehicle Charging Demand." Transportation Research Record / USDOT ROSAP. https://rosap.ntl.bts.gov/view/dot/60864
[3] Lamontagne, S., Carvalho, M., Frejinger, E., & Atallah, R. (2025). "What makes a good public EV charging station? A revealed preference study." arXiv:2504.17722. https://arxiv.org/abs/2504.17722
[4] Molin, E., et al. (2025). "Modeling EV charging behavior with hybrid choice models." Transportation Research Part A. DOI: 10.1016/j.tra.2025.
[5] Dong, L., Hardman, S., & Bunch, D. S. (2024). "Cost Sensitivity and Charging Choices of Plug-in Electric Vehicle Drivers - A Stated Preference Study." National Center for Sustainable Transportation, NCST-UCD-RR-24-33. DOI: 10.7922/G2NP22S0.
[6] Wu, H., et al. (2026). "Agent-Based Modeling to Evaluate the Potential of a Partial-Charging Strategy." MDPI World Electric Vehicle Journal.
[7] Tran, T., Zhao, L., & Xiong, L. (2026). "TrajGenAgent: A Hierarchical LLM Agent for Human Mobility Trajectory Generation." arXiv:2606.12657. https://arxiv.org/abs/2606.12657
[8] Park, J. S., et al. (2024). "LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals." arXiv:2411.10109. https://arxiv.org/abs/2411.10109
[9] Sun, Y., et al. (2026). "PEMANT: Persona-Enriched Multi-Agent Negotiation for Travel." arXiv:2604.10475. https://arxiv.org/abs/2604.10475
[10] Hörl, S., & Balac, M. (2021). "Synthetic population and travel demand for Paris and Île-de-France based on an open and extendable pipeline." Transportation Letters. DOI: 10.1080/21681376.2021.1968941.
[11] Liu, Y., Liao, X., Ma, H., He, B. Y., Stanford, C., & Ma, J. (2024). "Human Mobility Modeling with Household Coordination Activities under Limited Information via Retrieval-Augmented LLMs." arXiv:2409.17495. https://arxiv.org/abs/2409.17495
[12] Potoglou, D., Song, R., & Santos, G. (2023). "Public charging choices of electric vehicle users: A review and conceptual framework." Transportation Research Part D, 121, 103824. DOI: 10.1016/j.trd.2023.103824.
[13] Li, Q., Ji, C., & Liu, X. (2025). "From Narrative to Action: A Hierarchical LLM-Agent Framework for Human Mobility Generation." arXiv:2510.24802. https://arxiv.org/abs/2510.24802
[14] Liu, T., Yang, J., & Yin, Y. (2024). "Toward LLM-Agent-Based Modeling of Transportation Systems: A Conceptual Framework." arXiv:2412.06681. https://arxiv.org/abs/2412.06681
[15] Xiao, P., Li, Y., Mukhopadhyay, A., et al. (2026). "Planning, Scheduling, and Behavior in EV Charging Systems: A Critical Survey and Trilemma Framework." arXiv:2605.21665. https://arxiv.org/abs/2605.21665
[16] Metropolitan Washington Council of Governments (MWCOG) (2026). "2025 State of the Commute Survey Report."
"""

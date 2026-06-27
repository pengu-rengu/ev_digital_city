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
        "A relative calls midday proposing an outing; you're parked at a store with a low battery and modest range. (A) Top up now at the cheap L2 here while parked, even if you have to wait, (B) pay for a quick DC so you're free to leave the moment plans firm up, or (C) head out on your current range and charge opportunistically wherever the outing lands you.",
        "You're at 80%, charging has slowed, and a longer family-visit drive is likely tomorrow. (A) Unplug at 80% now and deal with tomorrow tomorrow, (B) wait for near-full so the longer drive has a comfortable buffer, or (C) unplug now but pre-plan a charging stop on tomorrow's route so you're covered without waiting.",
        "A free L2 charger is 10 minutes farther; a paid DC sits right on your route; your day is flexible. (A) Detour for the free L2 and run an errand nearby while it charges, (B) take the quick paid DC on route to keep the day easy, or (C) skip charging since local range is fine and decide later as the day unfolds."
    ],
    Archetype.PARENT_COMMUTER: [
        "Mid-charge at a slow, cheap station, school calls for an early pickup and your battery is still low. (A) Abandon the charge and go straight to pickup, (B) move to a DC station near the school for a fast top-up on the way, or (C) ask your partner to cover pickup so you can finish the cheap charge.",
        "Your partner needs the shared car by 6 PM; you're low with errands left. (A) Fast DC now to hand it back full and on time, (B) cheap slow charge and drop a non-essential errand, or (C) skip charging, finish the kid errands, and hand it back low.",
        "Three kid runs plus groceries today. (A) One tight loop with a fast DC mid-way so you're never caught low with kids aboard, (B) one loop, charging slow during the longest activity to save money, or (C) separate trips as each comes up, grabbing a charge only if a pickup forces a wait."
    ],
    Archetype.FLEXIBLE_COMMUTER: [
        "It's a remote day, public charging prices spike midday, and your battery is moderate. (A) Wait and charge in the cheap off-peak evening window, (B) don't charge at all today, let it sit and top up cheap another day, or (C) charge now anyway, mid-errand, since you're already out and it's convenient.",
        "A friend proposes a spontaneous cross-town dinner after today's trips; your range is tightish. (A) Go, and fast-charge en route first to be safe, (B) go only if your current range covers it and skip charging, or (C) go and charge opportunistically wherever you end up near the restaurant.",
        "Your errands could be done now around the office area or deferred to a remote day. (A) Bundle them now and charge at the longest stop during peak, (B) defer to a low-traffic remote day and charge cheap off-peak, or (C) do them now, adding a couple impromptu stops and charging wherever convenient."
    ],
    Archetype.RIGID_COMMUTER: [
        "It's a normal workday, your regular charger is unexpectedly offline, and you're at 40% with a fixed start time. (A) Pay more for a DC fast-charge at a station just off your route to stay topped up and on time, (B) drive on and run the day at 40%, betting your usual charger is back tomorrow, or (C) take a longer detour to another charger you know and trust, even if it costs you time.",
        "Charging prices jumped at your on-route station this month; a cheaper one sits 12 minutes off-route. (A) Eat the higher price to keep your exact routine and timing, (B) switch to the cheaper off-route charger and rebuild your routine around it, or (C) add an earlier cheap top-up so you keep a high charge buffer while managing the cost.",
        "You arrive and your usual charger has a two-car queue with uncertain wait; you have slack but not much. (A) Wait in line for your reliable usual charger, (B) leave immediately for a pricier DC station with open stalls to guarantee your timing, or (C) skip charging now since you have enough range and top up later at your cheap spot."
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
NON_COMMUTER scenarios (option -> disposition owners):
  1 midday outing call, low battery .. [7][8]   A:0,2 B:1 C:3
  2 80% taper before long drive ...... [6]       A:0,3 B:2 C:1
  3 free L2 detour vs paid DC on route [3][5]    A:0,2 B:1 C:3

PARENT_COMMUTER dispositions:
  Childcare-first juggler ............ [9][11]
  Shared-vehicle coordinator ......... [10][11]
  Budget-stretched family driver ..... [5][12]
  Time-pressed convenience parent .... [2][4]
PARENT_COMMUTER scenarios (option -> disposition owners):
  1 early pickup mid-charge .......... [9][11]   A:0 B:3 C:1,2
  2 partner needs car by 6 PM ........ [10]       A:1,3 B:2 C:0
  3 three kid runs + groceries ....... [12][13]   A:1,3 B:2 C:0

FLEXIBLE_COMMUTER dispositions:
  Off-peak optimizer ................. [4][16]
  Spontaneous remote-day runner ...... [7][13]
  Office-day convenience charger ..... [4][16]
  Frugal homebody hybrid ............. [2][5]
FLEXIBLE_COMMUTER scenarios (option -> disposition owners):
  1 price spike on a remote day ...... [4]        A:0 B:3 C:1,2
  2 spontaneous cross-town dinner .... [7][8]      A:2 B:0,3 C:1
  3 bundle today vs defer ............ [12][13]    A:2 B:0,3 C:1

RIGID_COMMUTER dispositions:
  Schedule-protecting fast-charger ... [13][14]
  Routine cost-minimizer ............. [3][5]
  Range-anxious commuter ............. [2]
  Corridor-dependent charger ......... [15]
RIGID_COMMUTER scenarios (option -> disposition owners):
  1 usual charger offline, at 40% .... [14][15]   A:0,2 B:1 C:3
  2 price jump, cheaper off-route .... [3][5]      A:0,3 B:1 C:2
  3 two-car queue, uncertain wait .... [14][15]    A:3 B:0,2 C:1

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

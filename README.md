# EV Digital City

Agent-based simulation of electric-vehicle daily travel and charger contention in Reston, VA.
The pipeline turns a regional travel survey (RTS) plus OpenStreetMap and EV-charger data into
LLM-driven driver agents that each plan a full day, then simulates competition for charging ports.

## Architecture

The project is a staged pipeline. Each script in `src/` is one stage; it reads and writes JSON
artifacts in `artifacts/`. Stages run in dependency order:

```
data/*.csv ──────────────────────────────┐
                                          ▼
taz_centroids.py ─► *_taz_centroids.json ─► profiles.py ─► profiles.json ─┐
                                                                          ├─► personas.py ─► personas.json ─┐
*.osm.pbf + alt_fuel_stations.csv ─► nodes.py ─► nodes.json ──────────────┼─────────────────────────────────┤
                                                                          │                                 ▼
roads.py ─► roads.json ────────────────────────────────────────────────────┘            agent.py ─► agents.json
                                                                                                          │
nodes.json + agents.json ─► simulation.py ─► simulation_logs.json  ◄──────────────────────────────────────┘
```

`filter_profiles.py` (→ `filtered_profiles.json`) and `combined_map.py` (→ `combined_map.png`) are
side branches used by the notebooks, not by the simulation.

### `src/` modules

| Module | Reads | Writes | Purpose |
|---|---|---|---|
| `taz_centroids.py` | TPB & BMC ArcGIS services | `tpb_taz_centroids.json`, `bmc_taz_centroids.json` | Fetch traffic-analysis-zone polygons, compute centroid coords used to geolocate survey trips. |
| `nodes.py` | `data/*.osm.pbf`, `data/alt_fuel_stations.csv`, TIGER place boundary | `nodes.json`, `nodes_map.png` | Extract Reston destinations (houses, offices, shops, …) from OSM and EV chargers from AFDC; attach chargers to nearby nodes. |
| `roads.py` | VDOT speed-limit ArcGIS service | `roads.json`, `roads_map.png` | Fetch Reston road segments with posted speed limits (used for travel-time routing). |
| `combined_map.py` | `nodes.json`, `roads.json` | `combined_map.png` | Overlay nodes and roads on one Reston map. |
| `profiles.py` | `data/person.csv`, `household.csv`, `vehicle.csv`, `trip.csv`, centroid JSONs | `profiles.json` | Build per-person `Profile`s: demographics, attributes (caregiver, mobility, work arrangement, schedule irregularity), commuter archetype, and geolocated trips. |
| `filter_profiles.py` | `profiles.json` | `filtered_profiles.json` | Naive-Bayes filter that keeps profiles likely to be EV owners (notebooks only). |
| `personas.py` | `profiles.json` | `personas.json` | For an all-EV target profile, iteratively generate → score → reflect → refine a natural-language persona with the LLM. |
| `agent.py` | `personas.json`, `nodes.json`, `roads.json` | `agents.json` | LLM agent role-plays the persona and builds a daily schedule (stops + one charge) using search / A* routing / schedule tools. |
| `simulation.py` | `agents.json`, `nodes.json` | `simulation_logs.json` | Detect charger contention across agents and let each agent resolve it (queue / relocate / give up); emit timeline + contention events. |
| `labels.py` | — | — | Survey integer-code → human-label lookup dicts used by `profiles.py`. |

### `notebooks/`

| Notebook | Purpose |
|---|---|
| `initial_analysis.ipynb` | Early scoping: state/county FIPS exploration, choosing Fairfax County / Reston as the study area. |
| `profile_validation.ipynb` | Compare distributions (income, employment, age, arrival times) of filtered vs EV profiles. |
| `archetype_validation.ipynb` | Sanity-check the archetype classifier (caregiver share, non-work/home trips, drop-off trips, work arrangement per archetype). |
| `mobility_analysis.ipynb` | Interactive trip-trajectory maps, destination heatmaps, and origin-destination heatmaps per archetype. |
| `clustering.ipynb` | Encode/scale person-level features, run UMAP + clustering on filtered profiles. |
| `old/` | Deprecated earlier versions (`only_ev_clustering`, `od_analysis`). |

## Setup

Requires Python (see `.python-version`), git, and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/pengu-rengu/ev_digital_city.git
cd ev_digital_city
uv sync
```

The persona / agent / simulation stages call the OpenAI API. Create `src/.env`:

```bash
echo "OPENAI_API_KEY=sk-..." > src/.env
```

`artifacts/` and the OSM extract are not committed and must be generated/downloaded (next section).

## Generating artifacts and running the simulation

Run every command from the repo root. `artifacts/` is created automatically.

**1. Download the OSM extract** (gitignored) into `data/` — Virginia from Geofabrik
(https://download.geofabrik.de/north-america/us/virginia.html), saved as
`data/virginia-260608.osm.pbf`.

**2. Generate the geographic and profile artifacts** (these hit public ArcGIS/TIGER services and the
local data files; no API key needed):

```bash
uv run src/taz_centroids.py   # -> tpb/bmc_taz_centroids.json
uv run src/nodes.py           # -> nodes.json, nodes_map.png
uv run src/roads.py           # -> roads.json, roads_map.png
uv run src/profiles.py        # -> profiles.json
```

**3. Generate the LLM-driven artifacts** (requires `OPENAI_API_KEY` in `src/.env`):

```bash
uv run src/personas.py        # -> personas.json   (from profiles.json)
uv run src/agent.py           # -> agents.json     (from personas.json, nodes.json, roads.json)
uv run src/simulation.py      # -> simulation_logs.json  (from agents.json, nodes.json)
```

`simulation_logs.json` is the final output: the agent status timeline plus charger-contention events
and the agents' reasoning traces.

Optional extras:

```bash
uv run src/combined_map.py    # -> combined_map.png
uv run src/filter_profiles.py # -> filtered_profiles.json (needed by the notebooks below)
```

## Running the notebooks

The analysis/validation notebooks read the artifacts above (`filter_profiles.py` must have been run
for the ones that use `filtered_profiles.json`). Start Jupyter and open them from the URL it prints:

```bash
uv run jupyter lab
```

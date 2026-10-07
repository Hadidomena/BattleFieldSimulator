# BattleFieldSimulator

A meso-scale tactical combat simulator built with [Mesa](https://mesa.readthedocs.io/) agent-based modeling framework. Created as part of an Engineering degree in Applied Computer Science.

## Features

### Unit classes
- **InfantrySquad** — balanced all-rounder
- **ReconSquad** — high mobility, wide vision (180°), low HP, early retreat (40%), cover-seeking AI
- **MechanizedInfantry** — armoured transport with good firepower
- **MainBattleTank** — heavy armour (0.60), massive HP (400), long-range cannon

### Tactical AI
Modeled using finite-state machine utilizing states:
- **advance** — move toward enemies using A*/Dijkstra pathfinding
- **engage** — attack visible enemies in range while repositioning to nearby cover that keeps the target in range and in line of sight; re-seeks cover when the current cover is destroyed
- **retreat** — scored-cell selection (distance, LoS breaks, cover, group cohesion), covering fire, group coordination, non-abandonment check
- **patrol** — follow a predefined route, engage enemies on sight

### Combat mechanics
- **Hit probability**: distance penalty, cover penalty, stability bonus from attacker cover
- **Damage model**: range falloff, cover reduction, armour penetration, **flanking bonus (+50% outside vision cone)**
- **Line of Sight**: Bresenham's algorithm with terrain obstacles
- **Detection**: vision cone + LoS + range threshold per unit class
- **Directional cover**: only obstacles between defender and attacker count
- **Per-class cover multiplier**: Infantry/Recon 1.0, Mechanized 0.7, MBT 0.3
- **Destructible cover**: obstacles have HP, take splash damage (50%) from hits, crumble when depleted

### Telemetry & analysis
- `TelemetryCollector` — records model state, agent state, and combat events per step
- Export of per scenario data to CSV and JSON
- `ScenarioAnalyzer` — 10 aggregate comparison charts
- `ScenarioExplorer` — 6 per-scenario deep-dive charts (population, damage, AI states, events, HP trajectories)
- Most of tweaks and fixes were based on the analysis of the results of 10 test scenarios created to check behaviour in different situations.

### Scenario runner
- JSON-defined experiments with map, unit config, overrides, and repetitions
- 10 pre-built scenarios across 5 maps testing: unit balance, cover, detection, chokepoints, numerical superiority, mobility vs durability, retreat thresholds
- Automated aggregation (win rates, KDR, DPS, survival) and telemetry export

### Interactive GUI
- Browser-based UI built on Mesa's Solara integration
- Live battlefield with a unit/team/AI-state legend and an optional AI-state outline overlay
- Live charts: population, damage, destroyed obstacles, AI-state distribution
- Config editor (scenario, map, unit classes/counts, seed, step cap) that rebuilds instantly
- Record a live run, then replay saved telemetry with a step scrubber

## Getting started

```bash
pip install -r requirements.txt
```

### Run a simulation

```bash
python main.py                                           # 10x10 board, 1v1
python main.py --steps 20                                # fixed steps
python main.py --seed 42                                 # reproducible run
python main.py --map data/example_large_map.csv --blue-units 3 --red-units 3
```

### Launch the interactive GUI

An interactive browser-based visualization of a live simulation is built with
Mesa's Solara integration. The first page load can take a while while the
Solara server compiles the app.

```bash
python -m simulation.gui            # launches the Solara server
```

The GUI shows the battlefield map, live charts (population, damage, destroyed
obstacles, AI-state distribution), a unit/team/AI-state legend, and playback
controls (play/pause, step, reset). An optional overlay outlines each unit by
its current AI state (advance/engage/retreat/patrol). The **Model Parameters**
panel lets you configure the run: choose a preset scenario, or pick a map plus
per-team unit class and count, set a seed, and cap the number of steps.
Changing a setting rebuilds the simulation immediately.

Use **Record run** to save the current live run's telemetry, then switch to the
**Replay** view to scrub through saved runs (from `run_scenarios.py` or recorded
from the GUI): pick a scenario/run, drag the step slider, and inspect the
reconstructed battlefield, population chart, and per-step statistics.

### Run experimental scenarios

```bash
python run_scenarios.py                                  # all 10 scenarios
python run_scenarios.py --scenario scenario_01           # specific scenario
python run_scenarios.py --seed 42                        # reproducible runs
python run_scenarios.py --list                           # list available
python run_scenarios.py --no-telemetry                   # skip telemetry export
```

### Generate analysis charts
Charts are generated using data from scenarios, so running them beforehand is needed.
```bash
python -m simulation.analysis                            # aggregate charts
python -m simulation.analysis --explore                  # per-scenario charts
python -m simulation.analysis --explore "H1"             # specific scenario
```

### Run tests

```bash
pytest -v --cov=simulation
```

### Set up git hooks

Git hooks are managed with [Lefthook](https://lefthook.dev/). They run ruff
linting/formatting and byte-compile staged Python files before each commit.

```bash
winget install lefthook          # if using Windows
sudo apt install lefthook        # or
sudo snap install lefthook       # if using Linux
brew install lefthook            # if on MacOS
# in case of problems see https://lefthook.dev/installation/

lefthook install         # one-time, per clone
```

## Project structure

```
simulation/
├── agent.py          # CombatAgent and its subclasses
├── model.py          # BattlefieldModel (Mesa Model), obstacle HP tracking
├── utils.py          # Pathfinding, LoS, cover, hit/damage math, detection
├── scenarios.py      # Scenario loading + shared model builder
├── telemetry.py      # Data recording and CSV/JSON export
├── analysis.py       # ScenarioAnalyzer + ScenarioExplorer (charts)
├── gui/              # Solara GUI
│   ├── app.py             # Page layout: live view, charts, replay view
│   ├── model_factory.py   # SimulationModel + config-editor parameters
│   ├── portrayal.py       # Agent/terrain drawing for the map
│   ├── theme.py           # Colors/markers for teams, classes, AI states
│   └── replay.py          # Telemetry loading/recording for the replay view
└── tests/            # Unit tests (pytest)

data/scenarios/
├── maps/             # CSV map grids (0=open, 1=obstacle)
├── scenario_*.json   # Experiment definitions
├── results/          # Telemetry output (per run, used by the replay view)
└── analysis/         # Generated chart images
```

## Maps

Map files are CSV grids where `0` = open terrain (cost 1.0) and `1` = obstacle (impassable, blocks LoS, provides cover). Other positive values represent rough terrain with elevated movement cost. Obstacles can be destroyed by sustained fire (destructible cover).

from __future__ import annotations

import json
from pathlib import Path

from simulation.agent import (
	CombatAgent,
	InfantrySquad,
	MainBattleTank,
	MechanizedInfantry,
	ReconSquad,
)
from simulation.model import BattlefieldModel
from simulation.utils import is_walkable, load_board

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = PROJECT_ROOT / "data" / "scenarios"
MAP_DIR = SCENARIO_DIR / "maps"

UNIT_CLASSES: dict[str, type[CombatAgent]] = {
	"CombatAgent": CombatAgent,
	"InfantrySquad": InfantrySquad,
	"ReconSquad": ReconSquad,
	"MechanizedInfantry": MechanizedInfantry,
	"MainBattleTank": MainBattleTank,
}


def load_scenarios(
	scenario_dir: Path = SCENARIO_DIR, filters: list[str] | None = None
) -> list[dict]:
	scenarios: list[dict] = []
	for filepath in sorted(scenario_dir.glob("scenario_*.json")):
		if filters:
			if not any(f in filepath.stem for f in filters):
				continue
		with open(filepath) as f:
			scenario = json.load(f)
		scenario["_filename"] = filepath.stem
		scenarios.append(scenario)
	return scenarios


def default_spawn_points(board, team: str, count: int) -> list[tuple[int, int]]:
	width = board.shape[1]
	height = board.shape[0]
	positions: list[tuple[int, int]] = []
	if count <= 0:
		return positions

	if team == "Blue":
		candidate_rows = [1, 2, 3]
		candidate_columns = list(range(1, min(width - 1, count + 2)))
	else:
		candidate_rows = [height - 2, height - 3, height - 4]
		candidate_columns = list(range(max(1, width - count - 2), width - 1))

	for row in candidate_rows:
		for column in candidate_columns:
			if len(positions) >= count:
				return positions
			if is_walkable((column, row), board):
				positions.append((column, row))

	return positions


def list_maps(map_dir: Path = MAP_DIR) -> list[Path]:
	maps = sorted(map_dir.glob("*.csv"))
	extra = PROJECT_ROOT / "data" / "example_large_map.csv"
	if extra.exists():
		maps.append(extra)
	return maps


def build_model_from_scenario(
	scenario: dict, seed: int | None = None
) -> BattlefieldModel:
	board = load_board(scenario["map"], base_dir=PROJECT_ROOT)
	blue_config = scenario["blue_team"]
	red_config = scenario["red_team"]

	blue_spawn = [tuple(sp) for sp in blue_config.get("spawn_points", [])]
	red_spawn = [tuple(sp) for sp in red_config.get("spawn_points", [])]

	model = BattlefieldModel(
		board,
		blue_spawn_points=blue_spawn,
		red_spawn_points=red_spawn,
		seed=seed,
		spawn_agents=False,
	)

	configure_agents(model, "Blue", blue_config)
	configure_agents(model, "Red", red_config)

	model.telemetry.reset()
	model.telemetry.record_step(model)
	return model


def configure_agents(model: BattlefieldModel, team: str, team_config: dict) -> None:
	to_remove = [a for a in model.agents if getattr(a, "team", None) == team]

	for agent in to_remove:
		if agent.pos is not None:
			model.grid.remove_agent(agent)
		agent.remove()

	unit_defs = team_config.get("units", [])
	spawn_points = team_config.get("spawn_points", [])

	spawn_index = 0
	for unit_def in unit_defs:
		class_name = unit_def["class"]
		count = unit_def.get("count", 1)
		overrides = unit_def.get("overrides", {})

		agent_cls = UNIT_CLASSES[class_name]

		for _ in range(count):
			spawn_pos = (
				tuple(spawn_points[spawn_index])
				if spawn_index < len(spawn_points)
				else None
			)
			spawn_index += 1

			if spawn_pos is None:
				continue

			agent = agent_cls(model, team=team, **overrides)
			model.grid.place_agent(agent, spawn_pos)

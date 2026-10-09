from __future__ import annotations

from pathlib import Path

from simulation.agent import CombatAgent
from simulation.model import BattlefieldModel
from simulation.scenarios import (
	MAP_DIR,
	PROJECT_ROOT,
	UNIT_CLASSES,
	configure_agents,
	default_spawn_points,
	load_scenario,
)
from simulation.utils import load_board

DEFAULT_MAP = "open_field_10x10.csv"
DEFAULT_UNIT_CLASS = "InfantrySquad"
CUSTOM_SCENARIO = "Custom"


def _resolve_map(map_name: str | Path) -> Path:
	path = Path(map_name)
	candidates = (path, PROJECT_ROOT / "data" / path, MAP_DIR / path)
	for candidate in candidates:
		if candidate.is_file():
			return candidate

	raise FileNotFoundError(f"Map not found: {map_name}")


def _resolve_unit_class(class_name: str) -> type[CombatAgent]:
	try:
		return UNIT_CLASSES[class_name]
	except KeyError:
		raise ValueError(
			f"Unknown unit class: {class_name}. "
			f"Available: {', '.join(sorted(UNIT_CLASSES))}"
		) from None


def _parse_seed(seed: str | int | None) -> int | None:
	if seed is None or seed == "":
		return None
	try:
		return int(seed)
	except (TypeError, ValueError):
		return None


class SimulationModel(BattlefieldModel):
	"""BattlefieldModel with keyword-only, defaulted parameters.

	Mesa's visualization inspects the model's ``__init__`` signature to build
	its parameter controls and to recreate the model on reset. ``BattlefieldModel``
	requires a positional ``board`` array, which cannot be expressed that way,
	so this subclass builds the board internally, either from a named scenario
	or from a map plus per-team unit configuration.
	"""

	def __init__(
		self,
		scenario_name: str = CUSTOM_SCENARIO,
		map_name: str = DEFAULT_MAP,
		blue_units: str = DEFAULT_UNIT_CLASS,
		red_units: str = DEFAULT_UNIT_CLASS,
		blue_count: int = 3,
		red_count: int = 3,
		seed: str | int | None = None,
		max_steps: int = 0,
	) -> None:
		resolved_seed = _parse_seed(seed)

		if scenario_name and scenario_name != CUSTOM_SCENARIO:
			scenario = load_scenario(scenario_name)
			board = load_board(scenario["map"], base_dir=PROJECT_ROOT)
			blue_config = scenario["blue_team"]
			red_config = scenario["red_team"]

			super().__init__(
				board,
				blue_spawn_points=[
					tuple(point) for point in blue_config.get("spawn_points", [])
				],
				red_spawn_points=[
					tuple(point) for point in red_config.get("spawn_points", [])
				],
				seed=resolved_seed,
				spawn_agents=False,
			)

			configure_agents(self, "Blue", blue_config)
			configure_agents(self, "Red", red_config)
			self.telemetry.reset()
			self.telemetry.record_step(self)
		else:
			board = load_board(_resolve_map(map_name))
			super().__init__(
				board,
				blue_unit_class=_resolve_unit_class(blue_units),
				red_unit_class=_resolve_unit_class(red_units),
				blue_spawn_points=default_spawn_points(board, "Blue", blue_count),
				red_spawn_points=default_spawn_points(board, "Red", red_count),
				seed=resolved_seed,
			)

		self.max_steps = max_steps if max_steps and max_steps > 0 else None

	def step(self) -> None:
		super().step()
		if self.max_steps is not None and self.steps >= self.max_steps:
			self.running = False


def unit_class_names() -> list[str]:
	return sorted(UNIT_CLASSES)


def default_model_parameters() -> dict:
	return {
		"scenario_name": CUSTOM_SCENARIO,
		"map_name": DEFAULT_MAP,
		"blue_units": DEFAULT_UNIT_CLASS,
		"red_units": DEFAULT_UNIT_CLASS,
		"blue_count": 3,
		"red_count": 3,
		"seed": "",
		"max_steps": 0,
	}


def build_model(**kwargs) -> SimulationModel:
	return SimulationModel(**kwargs)

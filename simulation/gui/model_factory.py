from __future__ import annotations

from pathlib import Path

from simulation.agent import CombatAgent
from simulation.model import BattlefieldModel
from simulation.scenarios import MAP_DIR, UNIT_CLASSES, default_spawn_points
from simulation.utils import load_board

DEFAULT_MAP = "open_field_10x10.csv"
DEFAULT_UNIT_CLASS = "InfantrySquad"


def _resolve_map(map_name: str | Path) -> Path:
	path = Path(map_name)
	if path.is_file():
		return path

	candidate = MAP_DIR / path
	if candidate.is_file():
		return candidate

	raise FileNotFoundError(f"Map not found: {map_name} (looked in {MAP_DIR})")


def _resolve_unit_class(class_name: str) -> type[CombatAgent]:
	try:
		return UNIT_CLASSES[class_name]
	except KeyError:
		raise ValueError(
			f"Unknown unit class: {class_name}. "
			f"Available: {', '.join(sorted(UNIT_CLASSES))}"
		) from None


class SimulationModel(BattlefieldModel):
	def __init__(
		self,
		map_name: str = DEFAULT_MAP,
		blue_units: str = DEFAULT_UNIT_CLASS,
		red_units: str = DEFAULT_UNIT_CLASS,
		blue_count: int = 3,
		red_count: int = 3,
		seed: int | None = None,
	) -> None:
		board = load_board(_resolve_map(map_name))

		super().__init__(
			board,
			blue_unit_class=_resolve_unit_class(blue_units),
			red_unit_class=_resolve_unit_class(red_units),
			blue_spawn_points=default_spawn_points(board, "Blue", blue_count),
			red_spawn_points=default_spawn_points(board, "Red", red_count),
			seed=seed,
		)


def build_model(**kwargs) -> SimulationModel:
	return SimulationModel(**kwargs)

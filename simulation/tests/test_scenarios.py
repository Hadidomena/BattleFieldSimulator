import numpy as np
import pytest

from simulation.model import BattlefieldModel
from simulation.scenarios import (
	build_model_from_scenario,
	configure_agents,
	default_spawn_points,
	list_map_names,
	list_scenario_names,
	load_scenario,
	load_scenarios,
)
from simulation.utils import is_walkable


def test_load_scenarios_returns_all_with_filenames() -> None:
	scenarios = load_scenarios()

	assert len(scenarios) == 10
	assert all("_filename" in scenario for scenario in scenarios)
	assert all("name" in scenario for scenario in scenarios)


def test_list_map_names_includes_known_maps() -> None:
	names = list_map_names()

	assert "open_field_10x10.csv" in names
	assert "example_large_map.csv" in names


def test_list_scenario_names_matches_loaded_scenarios() -> None:
	names = list_scenario_names()

	assert len(names) == len(load_scenarios())
	assert any(name.startswith("scenario_01") for name in names)


def test_load_scenario_by_filename_and_unknown() -> None:
	scenario = load_scenario(list_scenario_names()[0])

	assert "map" in scenario
	assert "blue_team" in scenario

	with pytest.raises(ValueError, match="Unknown scenario"):
		load_scenario("does_not_exist")


def test_default_spawn_points_returns_requested_count() -> None:
	board = np.zeros((10, 10), dtype=int)

	blue = default_spawn_points(board, "Blue", 3)
	red = default_spawn_points(board, "Red", 2)

	assert len(blue) == 3
	assert len(red) == 2
	assert len(set(blue)) == 3
	assert all(is_walkable(position, board) for position in blue + red)


def test_default_spawn_points_zero_count_is_empty() -> None:
	board = np.zeros((10, 10), dtype=int)

	assert default_spawn_points(board, "Blue", 0) == []


def test_default_spawn_points_avoids_obstacles() -> None:
	board = np.zeros((10, 10), dtype=int)
	board[1, 1] = 1

	blue = default_spawn_points(board, "Blue", 1)

	assert blue != [(1, 1)]
	assert is_walkable(blue[0], board)


def test_default_spawn_points_caps_at_available_cells() -> None:
	board = np.zeros((10, 10), dtype=int)

	positions = default_spawn_points(board, "Blue", 100)

	assert 0 < len(positions) < 100
	assert len(set(positions)) == len(positions)
	assert all(is_walkable(position, board) for position in positions)


def test_build_model_from_scenario_configures_agents() -> None:
	scenario = load_scenario("scenario_01_unit_class_infantry_vs_recon")

	model = build_model_from_scenario(scenario, seed=1)

	classes = sorted(type(agent).__name__ for agent in model.agents)
	assert classes == ["InfantrySquad", "ReconSquad"]


def test_configure_agents_replaces_team_agents() -> None:
	board = np.zeros((10, 10), dtype=int)
	model = BattlefieldModel(
		board,
		blue_spawn_points=[(1, 1)],
		red_spawn_points=[(8, 8)],
		spawn_agents=False,
	)

	configure_agents(
		model,
		"Blue",
		{
			"units": [{"class": "MainBattleTank", "count": 2}],
			"spawn_points": [[2, 2], [3, 2]],
		},
	)

	blue_agents = model.agents_of_team("Blue")
	assert len(blue_agents) == 2
	assert all(type(agent).__name__ == "MainBattleTank" for agent in blue_agents)

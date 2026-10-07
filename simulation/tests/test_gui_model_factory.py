import pytest

from simulation.gui.model_factory import (
	SimulationModel,
	_parse_seed,
	_resolve_map,
	_resolve_unit_class,
	build_model,
	default_model_parameters,
	unit_class_names,
)
from simulation.scenarios import list_scenario_names


def test_default_parameters_build_custom_model() -> None:
	model = SimulationModel(**default_model_parameters())

	assert model.alive_count("Blue") == 3
	assert model.alive_count("Red") == 3


def test_custom_unit_classes_and_counts() -> None:
	model = SimulationModel(
		map_name="chokepoints_20x20.csv",
		blue_units="MainBattleTank",
		red_units="ReconSquad",
		blue_count=2,
		red_count=3,
		seed="42",
	)

	classes = sorted(type(agent).__name__ for agent in model.agents)
	assert classes == ["MainBattleTank", "MainBattleTank"] + ["ReconSquad"] * 3


def test_scenario_mode_loads_configured_agents() -> None:
	scenario_name = list_scenario_names()[0]
	model = SimulationModel(scenario_name=scenario_name)

	classes = sorted(type(agent).__name__ for agent in model.agents)
	assert len(classes) == 2
	assert "InfantrySquad" in classes
	assert "ReconSquad" in classes


def test_max_steps_stops_the_model() -> None:
	model = SimulationModel(max_steps=2)

	for _ in range(5):
		if model.running:
			model.step()

	assert model.steps == 2
	assert not model.running


def test_zero_max_steps_means_no_limit() -> None:
	model = SimulationModel(max_steps=0)

	assert model.max_steps is None


def test_scenario_mode_respects_max_steps() -> None:
	model = SimulationModel(scenario_name=list_scenario_names()[0], max_steps=3)

	for _ in range(10):
		if model.running:
			model.step()

	assert model.steps == 3
	assert not model.running


def test_unknown_scenario_raises() -> None:
	with pytest.raises(ValueError, match="Unknown scenario"):
		SimulationModel(scenario_name="does_not_exist")


def test_unit_class_names_lists_all_classes() -> None:
	names = unit_class_names()

	assert "InfantrySquad" in names
	assert "MainBattleTank" in names


def test_parse_seed_variants() -> None:
	assert _parse_seed("42") == 42
	assert _parse_seed(7) == 7
	assert _parse_seed("") is None
	assert _parse_seed(None) is None
	assert _parse_seed("not-a-number") is None


def test_resolve_map_by_name_and_missing() -> None:
	path = _resolve_map("open_field_10x10.csv")

	assert path.is_file()

	with pytest.raises(FileNotFoundError):
		_resolve_map("nope.csv")


def test_resolve_unit_class_unknown_raises() -> None:
	with pytest.raises(ValueError, match="Unknown unit class"):
		_resolve_unit_class("SpaceMarine")


def test_build_model_returns_simulation_model() -> None:
	model = build_model(blue_count=1, red_count=1)

	assert isinstance(model, SimulationModel)
	assert model.alive_count("Blue") == 1
	assert model.alive_count("Red") == 1


def test_simulation_model_missing_map_raises() -> None:
	with pytest.raises(FileNotFoundError):
		SimulationModel(map_name="does_not_exist.csv")

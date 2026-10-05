import pytest

from simulation.gui.model_factory import (
	SimulationModel,
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


def test_unknown_scenario_raises() -> None:
	with pytest.raises(ValueError, match="Unknown scenario"):
		SimulationModel(scenario_name="does_not_exist")


def test_unit_class_names_lists_all_classes() -> None:
	names = unit_class_names()

	assert "InfantrySquad" in names
	assert "MainBattleTank" in names

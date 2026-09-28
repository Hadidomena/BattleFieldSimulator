import numpy as np
import pytest

from simulation.agent import CombatAgent
from simulation.model import BattlefieldModel


@pytest.fixture
def basic_board() -> np.ndarray:
	"""Returns an example empty 10x10 board."""
	return np.zeros((10, 10), dtype=int)


def test_model_initialization(basic_board: np.ndarray) -> None:
	model = BattlefieldModel(basic_board)

	assert model.width == 10
	assert model.height == 10
	assert len(model.agents) == 2

	teams = [a.team for a in model.agents]
	assert "Blue" in teams
	assert "Red" in teams


def test_model_seed_makes_runs_reproducible(basic_board: np.ndarray) -> None:
	first = BattlefieldModel(basic_board, seed=42)
	second = BattlefieldModel(basic_board, seed=42)

	assert first.rng.random() == second.rng.random()
	assert first.random.random() == second.random.random()

	for _ in range(10):
		if not first.running and not second.running:
			break
		first.step()
		second.step()

	assert first.telemetry.model_records == second.telemetry.model_records
	assert first.telemetry.agent_records == second.telemetry.agent_records


def test_model_step(basic_board: np.ndarray) -> None:
	model = BattlefieldModel(basic_board)
	model.step()
	model.step()
	df = model.datacollector.get_model_vars_dataframe()
	assert len(df) == 2
	assert df["Alive_Blue"].iloc[0] == 1
	assert df["Alive_Red"].iloc[0] == 1


def test_datacollector_records_post_step_state(basic_board: np.ndarray) -> None:
	model = BattlefieldModel(basic_board)
	blue = next(a for a in model.agents if a.team == "Blue")
	red = next(a for a in model.agents if a.team == "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 2))

	blue.accuracy = 1.0
	blue.attack_range = 4.0
	blue.firepower = 50
	red.hp = 10

	model.step()

	df = model.datacollector.get_model_vars_dataframe()
	assert df["Alive_Red"].iloc[-1] == 0
	assert model.running is False


def test_model_collects_combat_stats(basic_board: np.ndarray) -> None:
	model = BattlefieldModel(basic_board)
	blue = next(
		a for a in model.agents if isinstance(a, CombatAgent) and a.team == "Blue"
	)
	red = next(a for a in model.agents if isinstance(a, CombatAgent) and a.team == "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 2))

	blue.accuracy = 1.0
	blue.attack_range = 4.0
	blue.firepower = 50
	red.hp = 10

	blue.attack(red)

	assert model.shots_by_team["Blue"] == 1
	assert model.hits_by_team["Blue"] == 1
	assert model.damage_by_team["Blue"] > 0
	assert model.kills_by_team["Blue"] == 1
	assert model.eliminated_by_team["Red"] == 1

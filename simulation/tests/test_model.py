import numpy as np
import pytest

from simulation.model import BattlefieldModel


@pytest.fixture
def basic_board() -> np.ndarray:
	"""Zwraca przykładową, pustą planszę 10x10."""
	return np.zeros((10, 10), dtype=int)


def test_model_initialization(basic_board: np.ndarray) -> None:
	model = BattlefieldModel(basic_board)

	assert model.width == 10
	assert model.height == 10
	assert len(model.agents) == 2

	teams = [a.team for a in model.agents]
	assert "Blue" in teams
	assert "Red" in teams


def test_model_step(basic_board: np.ndarray) -> None:
	model = BattlefieldModel(basic_board)
	model.step()
	model.step()
	df = model.datacollector.get_model_vars_dataframe()
	assert len(df) == 2
	assert df["Alive_Blue"].iloc[0] == 1
	assert df["Alive_Red"].iloc[0] == 1

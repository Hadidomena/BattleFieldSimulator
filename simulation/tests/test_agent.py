from unittest.mock import MagicMock

import mesa

from simulation.agent import CombatAgent


def test_combat_agent_initialization() -> None:
	model_mock = MagicMock(spec=mesa.Model)
	agent = CombatAgent(
		model=model_mock,
		team="Blue",
		hp=120,
		firepower=15,
		observation_range=6,
		mobility=3,
	)

	assert agent.team == "Blue"
	assert agent.hp == 120
	assert agent.firepower == 15
	assert agent.observation_range == 6
	assert agent.mobility == 3


def test_combat_agent_step_alive() -> None:
	model_mock = MagicMock()
	model_mock.steps = 1
	agent = CombatAgent(model=model_mock, team="Red", hp=100)
	agent.pos = (1, 1)
	agent.step()


def test_combat_agent_step_dead() -> None:
	model_mock = MagicMock()
	model_mock.steps = 1
	agent = CombatAgent(model=model_mock, team="Red", hp=0)
	agent.pos = (1, 1)
	agent.step()

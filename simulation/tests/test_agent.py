from unittest.mock import MagicMock

import mesa
import numpy as np

from simulation.agent import CombatAgent
from simulation.model import BattlefieldModel


def _agent_by_team(model: BattlefieldModel, team: str) -> CombatAgent:
	for agent in model.agents:
		if isinstance(agent, CombatAgent) and agent.team == team:
			return agent
	raise AssertionError(f"Agent for team {team} not found")


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
	assert agent.view_angle_deg == 120.0
	assert agent.detection_threshold == 0.15
	assert agent.facing_direction == (0, 1)
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


def test_get_visible_enemies_detects_enemy_in_vision_cone() -> None:
	board = np.zeros((10, 10), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 4))

	blue.facing_direction = (0, 1)
	blue.view_angle_deg = 90.0
	blue.observation_range = 5
	blue.detection_threshold = 0.1

	visible = blue.get_visible_enemies()
	assert red in visible


def test_get_visible_enemies_rejects_enemy_outside_vision_cone() -> None:
	board = np.zeros((10, 10), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (3, 3))
	model.grid.move_agent(red, (3, 1))

	blue.facing_direction = (0, 1)
	blue.view_angle_deg = 90.0
	blue.observation_range = 5
	blue.detection_threshold = 0.1

	visible = blue.get_visible_enemies()
	assert red not in visible


def test_get_visible_enemies_rejects_enemy_behind_obstacle() -> None:
	board = np.zeros((10, 10), dtype=int)
	board[3, 2] = 1
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 5))

	blue.facing_direction = (0, 1)
	blue.view_angle_deg = 120.0
	blue.observation_range = 6
	blue.detection_threshold = 0.1

	visible = blue.get_visible_enemies()
	assert red not in visible

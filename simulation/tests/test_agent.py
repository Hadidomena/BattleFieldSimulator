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
	assert agent.attack_range == 3.0
	assert agent.view_angle_deg == 120.0
	assert agent.detection_threshold == 0.15
	assert agent.accuracy == 0.75
	assert agent.armor == 0.10
	assert agent.facing_direction == (0, 1)
	assert agent.mobility == 3
	assert agent.navigation_algorithm == "a_star"
	assert agent.allow_diagonal_navigation is False


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


def test_move_uses_pathfinding_to_bypass_obstacle_wall() -> None:
	board = np.zeros((7, 7), dtype=int)
	board[2, 2] = 1
	board[3, 2] = 1
	board[4, 2] = 1
	board[5, 2] = 1

	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (1, 3))
	model.grid.move_agent(red, (5, 3))

	blue.mobility = 1
	blue.navigation_algorithm = "a_star"
	blue.allow_diagonal_navigation = False

	blue.move()

	assert blue.pos == (1, 2)
	assert blue.current_path[0] == (1, 3)
	assert blue.current_path[-1] == (5, 3)


def test_attack_reduces_enemy_hp() -> None:
	board = np.zeros((6, 6), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 2))

	blue.attack_range = 4.0
	blue.accuracy = 1.0
	blue.firepower = 18
	red.hp = 40

	hit = blue.attack(red)

	assert hit is True
	assert red.hp < 40
	assert model.damage_by_team["Blue"] > 0


def test_attack_can_eliminate_and_remove_agent() -> None:
	board = np.zeros((6, 6), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 2))

	blue.attack_range = 4.0
	blue.accuracy = 1.0
	blue.firepower = 50
	red.hp = 10

	hit = blue.attack(red)

	assert hit is True
	assert red.hp == 0
	assert red.pos is None
	assert red not in list(model.agents)
	assert model.eliminated_by_team["Red"] == 1
	assert model.kills_by_team["Blue"] == 1


def test_attack_with_accuracy_override() -> None:
	board = np.zeros((6, 6), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 2))

	blue.attack_range = 4.0
	blue.accuracy = 1.0
	blue.firepower = 10
	red.hp = 100

	hit = blue.attack(red, accuracy_override=0.0)
	assert hit is False
	assert red.hp == 100


def test_covering_fire_during_retreat() -> None:
	board = np.zeros((8, 8), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (3, 3))
	model.grid.move_agent(red, (3, 5))

	blue.attack_range = 5.0
	blue.accuracy = 1.0
	blue.max_hp = 100
	blue.hp = 10
	blue.retreat_health_ratio = 0.50
	red.hp = 100

	red.accuracy = 1.0
	red.firepower = 10
	red.attack_range = 5.0

	blue.step()

	assert blue.ai_state == "retreat"
	assert blue.last_damage_dealt > 0 or blue.hp < 10


def test_retreat_does_not_abandon_outnumbered_allies() -> None:
	board = np.zeros((10, 10), dtype=int)
	model = BattlefieldModel(board)

	blue = _agent_by_team(model, "Blue")
	blue_2 = CombatAgent(model, team="Blue", hp=60, firepower=10)
	model.grid.place_agent(blue_2, (3, 2))

	red = _agent_by_team(model, "Red")
	red_2 = CombatAgent(model, team="Red", hp=100, firepower=15)
	model.grid.place_agent(red_2, (3, 6))
	red_3 = CombatAgent(model, team="Red", hp=100, firepower=15)
	model.grid.place_agent(red_3, (5, 5))

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (3, 4))
	model.grid.move_agent(red_2, (3, 5))
	model.grid.move_agent(red_3, (5, 5))

	blue.attack_range = 4.0
	blue.accuracy = 1.0
	blue.max_hp = 100
	blue.hp = 20
	blue.retreat_health_ratio = 0.50

	blue.step()

	assert blue.ai_state == "engage", (
		f"Expected engage (outnumbered), got {blue.ai_state}"
	)


def test_retreat_group_coordination() -> None:
	board = np.zeros((16, 16), dtype=int)
	model = BattlefieldModel(board)

	blue = _agent_by_team(model, "Blue")
	blue_2 = CombatAgent(model, team="Blue", hp=40, firepower=10)
	model.grid.place_agent(blue_2, (2, 6))

	red = _agent_by_team(model, "Red")
	model.grid.place_agent(red, (15, 7))

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(blue_2, (2, 6))
	model.grid.move_agent(red, (15, 7))

	blue.attack_range = 8.0
	blue.observation_range = 20
	blue.accuracy = 1.0
	blue.hp = 30
	blue.max_hp = 100
	blue.retreat_health_ratio = 0.50
	blue.mobility = 3
	blue.view_angle_deg = 360.0
	blue.facing_direction = (1, 0)

	blue_2.attack_range = 8.0
	blue_2.observation_range = 20
	blue_2.accuracy = 1.0
	blue_2.hp = 25
	blue_2.max_hp = 100
	blue_2.retreat_health_ratio = 0.50
	blue_2.mobility = 3
	blue_2.view_angle_deg = 360.0
	blue_2.ai_state = "retreat"

	blue_before_pos = blue.pos
	blue.step()

	assert blue.ai_state == "retreat", f"Expected retreat, got {blue.ai_state}"

	assert blue.pos[0] < blue_before_pos[0], (
		f"Should move left (away from enemy at x=15): x {blue_before_pos[0]} -> {blue.pos[0]}"  # noqa
	)
	assert blue.pos[1] > blue_before_pos[1], (
		f"Group cohesion should pull toward centroid y=4: y {blue_before_pos[1]} -> {blue.pos[1]}"  # noqa
	)

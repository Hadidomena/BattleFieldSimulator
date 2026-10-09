from unittest.mock import MagicMock

import mesa
import numpy as np

from simulation.agent import CombatAgent, MechanizedInfantry, ReconSquad
from simulation.model import BattlefieldModel
from simulation.utils import (
	cover_ratio,
	euclidean_distance,
	has_line_of_sight,
)


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
	blue.last_known_enemy_pos = red.pos

	blue.move()

	assert blue.pos == (1, 2)
	assert blue.current_path[0] == (1, 3)
	assert blue.current_path[-1] == (5, 3)


def test_navigation_falls_back_to_map_center_without_detection() -> None:
	board = np.zeros((20, 20), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (1, 1))
	model.grid.move_agent(red, (18, 18))

	blue.view_angle_deg = 90.0
	blue.facing_direction = (0, -1)
	blue.observation_range = 3
	blue.detection_threshold = 0.1

	assert blue.get_visible_enemies() == []
	assert blue._get_navigation_target() == (10, 10)


def test_navigation_remembers_last_known_enemy_position() -> None:
	board = np.zeros((20, 20), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (5, 5))
	model.grid.move_agent(red, (6, 5))

	blue.view_angle_deg = 360.0
	blue.observation_range = 5
	assert red in blue.get_visible_enemies()
	seen_pos = red.pos

	model.grid.move_agent(red, (19, 19))
	blue.observation_range = 1

	assert blue.get_visible_enemies() == []
	assert blue._get_navigation_target() == seen_pos


def test_navigation_advance_target_avoids_blocked_center() -> None:
	board = np.zeros((5, 5), dtype=int)
	board[2, 2] = 1
	model = BattlefieldModel(
		board,
		blue_spawn_points=[(0, 0)],
		red_spawn_points=[(4, 4)],
	)
	blue = _agent_by_team(model, "Blue")

	blue.view_angle_deg = 90.0
	blue.facing_direction = (0, -1)
	blue.observation_range = 1

	assert blue.get_visible_enemies() == []
	target = blue._get_navigation_target()
	assert target is not None
	assert target != (2, 2)
	assert model.terrain[target[1], target[0]] == 0


def test_detection_alerts_allied_units() -> None:
	board = np.zeros((20, 20), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	blue_ally = CombatAgent(model, team="Blue", hp=100)
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (5, 5))
	model.grid.place_agent(blue_ally, (1, 1))
	model.grid.move_agent(red, (6, 5))

	blue.view_angle_deg = 360.0
	blue.observation_range = 5
	blue_ally.view_angle_deg = 90.0
	blue_ally.facing_direction = (0, -1)
	blue_ally.observation_range = 1

	assert blue_ally.last_known_enemy_pos is None
	blue.get_visible_enemies()
	assert blue_ally.last_known_enemy_pos == red.pos


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


def test_attack_damage_is_clamped_to_remaining_hp() -> None:
	board = np.zeros((6, 6), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 2))

	blue.attack_range = 4.0
	blue.accuracy = 1.0
	blue.firepower = 100
	red.hp = 5

	blue.attack(red)

	assert red.hp == 0
	assert model.damage_by_team["Blue"] == 5
	assert blue.last_damage_dealt == 5


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
	model.grid.move_agent(red, (3, 4))

	blue.attack_range = 5.0
	blue.accuracy = 1.0
	blue.max_hp = 100
	blue.hp = 20
	blue.retreat_health_ratio = 0.50
	blue.view_angle_deg = 360.0
	blue.firepower = 50
	red.hp = 100

	red.accuracy = 0.0
	red.firepower = 0
	red.attack_range = 0.0

	visible = blue.get_visible_enemies()
	assert len(visible) > 0, "Blue should see Red"

	shots_before = model.shots_by_team["Blue"]
	blue.step()
	shots_after = model.shots_by_team["Blue"]

	assert blue.ai_state == "retreat", f"Expected retreat, got {blue.ai_state}"
	assert shots_after > shots_before, (
		f"Covering fire should shoot at enemies (shots: {shots_before} -> {shots_after})"
	)


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


def test_allied_agents_filters_team_alive_and_radius() -> None:
	board = np.zeros((10, 10), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")

	ally = CombatAgent(model, team="Blue", hp=100)
	dead_ally = CombatAgent(model, team="Blue", hp=0)
	enemy = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (5, 5))
	model.grid.place_agent(ally, (5, 7))
	model.grid.place_agent(dead_ally, (5, 4))
	model.grid.move_agent(enemy, (6, 5))

	assert blue._allied_agents() == [ally]
	assert blue._allied_agents(radius=1) == []
	assert blue._allied_agents(radius=2) == [ally]
	assert blue._get_ally_positions() == [(5, 7)]

	ally.ai_state = "retreat"
	assert blue._allied_agents(state="retreat") == [ally]
	assert blue._allied_agents(state="engage") == []


def test_retreat_ignores_unreachable_cells() -> None:
	board = np.zeros((10, 10), dtype=int)
	for y in range(3):
		board[y, 3] = 1
	for x in range(3):
		board[3, x] = 1

	model = BattlefieldModel(
		board,
		blue_spawn_points=[(5, 5)],
		red_spawn_points=[(8, 8)],
	)
	blue = _agent_by_team(model, "Blue")

	assert (1, 1) not in blue._reachable_cells(model.terrain)

	blue.view_angle_deg = 360.0
	blue.observation_range = 20
	blue._score_retreat_cell = lambda candidate, *args: (
		100.0 if candidate == (1, 1) else 1.0
	)
	blue._move_randomly = MagicMock()

	blue.retreat(blue.get_visible_enemies())

	assert not blue._move_randomly.called


def test_destructible_cover_obstacle_destroyed_by_fire() -> None:
	board = np.zeros((6, 6), dtype=int)
	board[4, 3] = 1
	model = BattlefieldModel(board, obstacle_max_hp=20.0)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 4))

	blue.attack_range = 5.0
	blue.accuracy = 1.0
	blue.firepower = 30

	assert model.terrain[4, 3] == 1
	assert model.obstacle_hp[4, 3] == 20.0

	for _ in range(10):
		if red.hp <= 0:
			break
		blue.attack(red)

	assert model.obstacle_hp[4, 3] < 20.0, "Obstacle should take splash damage"


def test_destructible_cover_terrain_becomes_walkable() -> None:
	board = np.zeros((6, 6), dtype=int)
	board[4, 2] = 1
	model = BattlefieldModel(board, obstacle_max_hp=10.0)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 4))
	red.hp = 500

	blue.attack_range = 5.0
	blue.accuracy = 1.0
	blue.firepower = 50

	for _ in range(10):
		if red.hp <= 0:
			break
		blue.attack(red)

	assert model.terrain[4, 2] == 0, "Obstacle at target should be destroyed"
	assert model.destroyed_obstacles >= 1, "Destroyed obstacles should be counted"


def test_reconsquad_retreat_different_scoring() -> None:
	board = np.zeros((10, 10), dtype=int)
	board[5, 3] = 1
	board[5, 4] = 1
	board[5, 5] = 1
	model = BattlefieldModel(
		board,
		blue_unit_class=ReconSquad,
		red_unit_class=CombatAgent,
		blue_spawn_points=[(2, 2)],
		red_spawn_points=[(2, 6)],
	)
	recon = next(
		a for a in model.agents if isinstance(a, ReconSquad) and a.team == "Blue"
	)
	infantry = CombatAgent(model, team="Red", hp=100)

	enemy_positions = [(2, 8)]
	ally_positions = []
	group_centroid = None
	max_map_dist = 10.0

	test_pos = (5, 5)

	recon_score = recon._score_retreat_cell(
		test_pos, enemy_positions, ally_positions, group_centroid, max_map_dist
	)
	infantry_score = infantry._score_retreat_cell(
		test_pos, enemy_positions, ally_positions, group_centroid, max_map_dist
	)

	assert recon_score != infantry_score, (
		"ReconSquad should have different retreat scoring than base CombatAgent"
	)
	blue = next(
		a for a in model.agents if isinstance(a, ReconSquad) and a.team == "Blue"
	)
	red = next(a for a in model.agents if isinstance(a, CombatAgent) and a.team == "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 6))

	enemy_positions = [red.pos]
	ally_positions = []
	group_centroid = None
	max_map_dist = 10.0

	open_pos = (1, 2)
	cover_pos = (4, 5)

	open_score = blue._score_retreat_cell(
		open_pos, enemy_positions, ally_positions, group_centroid, max_map_dist
	)
	cover_score = blue._score_retreat_cell(
		cover_pos, enemy_positions, ally_positions, group_centroid, max_map_dist
	)

	assert cover_score > open_score, (
		f"ReconSquad should prefer cover positions "
		f"(cover={cover_score:.1f} vs open={open_score:.1f})"
	)


def test_reconsquad_earlier_retreat_threshold() -> None:
	recon = ReconSquad(MagicMock(spec=mesa.Model), team="Blue")
	infantry = CombatAgent(
		MagicMock(spec=mesa.Model), team="Blue", hp=100, retreat_health_ratio=0.30
	)

	assert recon.retreat_health_ratio == 0.40, (
		f"ReconSquad should retreat earlier (0.40), got {recon.retreat_health_ratio}"
	)
	assert infantry.retreat_health_ratio == 0.30, (
		f"Base units should retreat at 0.30, got {infantry.retreat_health_ratio}"
	)


def test_mechanized_infantry_accepts_retreat_health_ratio_override() -> None:
	default = MechanizedInfantry(MagicMock(spec=mesa.Model), team="Blue")
	overridden = MechanizedInfantry(
		MagicMock(spec=mesa.Model), team="Blue", retreat_health_ratio=0.55
	)

	assert default.retreat_health_ratio == 0.30, (
		f"MechanizedInfantry should default to 0.30, got {default.retreat_health_ratio}"
	)
	assert overridden.retreat_health_ratio == 0.55, (
		f"MechanizedInfantry should accept an override, got "
		f"{overridden.retreat_health_ratio}"
	)


def test_reconsquad_cover_seeking_advance() -> None:
	board = np.zeros((10, 10), dtype=int)
	board[4, 3] = 1
	board[4, 4] = 1
	board[4, 5] = 1
	model = BattlefieldModel(
		board,
		blue_unit_class=ReconSquad,
		red_unit_class=CombatAgent,
	)
	recon = next(
		a for a in model.agents if isinstance(a, ReconSquad) and a.team == "Blue"
	)
	red = next(a for a in model.agents if isinstance(a, CombatAgent) and a.team == "Red")

	model.grid.move_agent(recon, (2, 2))
	model.grid.move_agent(red, (2, 8))

	recon.ai_state = "advance"
	recon.mobility = 2
	recon.move()

	assert recon.pos is not None
	assert recon.pos != (2, 2), "ReconSquad should move from starting position"


def test_reconsquad_group_coordination() -> None:
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


def test_engage_moves_to_cover_instead_of_standing_in_the_open() -> None:
	board = np.zeros((12, 12), dtype=int)
	for x in range(3, 8):
		board[4, x] = 1

	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (5, 7))
	model.grid.move_agent(red, (8, 7))

	blue.view_angle_deg = 360.0
	blue.observation_range = 10
	blue.detection_threshold = 0.0
	blue.attack_range = 5.0
	blue.accuracy = 1.0
	blue.firepower = 20
	blue.mobility = 2
	red.hp = 100

	assert cover_ratio(blue.pos, board) == 0.0
	shots_before = model.shots_by_team["Blue"]

	for _ in range(4):
		blue.step()
		if cover_ratio(blue.pos, board) > 0.0:
			break

	assert blue.ai_state == "engage"
	assert cover_ratio(blue.pos, board) > 0.0, (
		f"Engaging unit should reposition to cover, stayed at {blue.pos}"
	)
	assert model.shots_by_team["Blue"] > shots_before, (
		"Unit should keep firing while repositioning"
	)


def test_engage_cover_cell_keeps_target_in_range_and_los() -> None:
	board = np.zeros((12, 12), dtype=int)
	board[5, 5] = 1

	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (3, 5))
	model.grid.move_agent(red, (7, 5))

	blue.view_angle_deg = 360.0
	blue.observation_range = 10
	blue.detection_threshold = 0.0
	blue.attack_range = 5.0
	blue.mobility = 2

	cell = blue._best_cover_cell(red)

	assert cell is not None
	assert euclidean_distance(cell, red.pos) <= blue.attack_range
	assert has_line_of_sight(cell, red.pos, board)


def test_unit_cannot_move_onto_occupied_allied_cell() -> None:
	board = np.zeros((8, 8), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	ally = CombatAgent(model, team="Blue", hp=100)

	model.grid.move_agent(blue, (2, 2))
	model.grid.place_agent(ally, (3, 2))

	blue.mobility = 2
	blue.move(target_position=(5, 2))

	assert blue.pos == (2, 2), "Unit must not stack onto an occupied cell"


def test_unit_cannot_move_onto_enemy_cell() -> None:
	board = np.zeros((8, 8), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (3, 2))

	blue.mobility = 2
	blue.move(target_position=(5, 2))

	assert blue.pos == (2, 2), "Unit must not enter a cell occupied by an enemy"


def test_unit_advances_while_path_stays_clear() -> None:
	board = np.zeros((8, 8), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")

	model.grid.move_agent(blue, (2, 2))

	blue.mobility = 2
	blue.move(target_position=(5, 2))

	assert blue.pos == (4, 2)

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import mesa
import numpy as np
import pandas as pd

import main as simulation_main
from simulation.agent import (
	CombatAgent,
	InfantrySquad,
	MainBattleTank,
	MechanizedInfantry,
	ReconSquad,
)
from simulation.model import BattlefieldModel
from simulation.utils import find_path


def _agent_by_team(model: BattlefieldModel, team: str) -> CombatAgent:
	for agent in model.agents:
		if isinstance(agent, CombatAgent) and agent.team == team:
			return agent
	raise AssertionError(f"Agent for team {team} not found")


def test_stage_i_object_architecture_defines_agent_model_and_environment() -> None:
	assert issubclass(CombatAgent, mesa.Agent)
	assert issubclass(BattlefieldModel, mesa.Model)

	board = np.zeros((10, 12), dtype=int)
	model = BattlefieldModel(board)
	assert model.width == 12
	assert model.height == 10
	assert isinstance(model.grid, mesa.space.MultiGrid)


def test_stage_i_turn_mechanism_activates_agents_once_per_model_step() -> None:
	board = np.zeros((8, 8), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	blue.step = MagicMock()
	red.step = MagicMock()

	model.step()

	blue.step.assert_called_once()
	red.step.assert_called_once()

	df = model.datacollector.get_model_vars_dataframe()
	assert len(df) == 1


def test_stage_i_main_loop_runs_declared_number_of_steps(monkeypatch) -> None:
	class FakeCollector:
		def get_model_vars_dataframe(self):
			return pd.DataFrame(
				[
					{
						"Alive_Blue": 1,
						"Alive_Red": 1,
						"Kills_Blue": 0,
						"Kills_Red": 0,
						"Damage_Blue": 0,
						"Damage_Red": 0,
					}
				]
			)

	class FakeBattlefieldModel:
		instances: list["FakeBattlefieldModel"] = []

		def __init__(
			self,
			board: np.ndarray,
			blue_spawn_points=None,
			red_spawn_points=None,
		) -> None:
			self.board = board
			self.step_calls = 0
			self.running = True
			self.datacollector = FakeCollector()
			self.blue_spawn_points = blue_spawn_points
			self.red_spawn_points = red_spawn_points
			FakeBattlefieldModel.instances.append(self)

		def step(self) -> None:
			self.step_calls += 1

	monkeypatch.setattr(simulation_main, "BattlefieldModel", FakeBattlefieldModel)
	monkeypatch.setattr(sys, "argv", ["main.py", "--steps", "3"])

	simulation_main.main()

	assert len(FakeBattlefieldModel.instances) == 1
	instance = FakeBattlefieldModel.instances[0]
	assert instance.step_calls == 3
	assert instance.board.shape == (10, 10)


def test_stage_i_environment_supports_basic_deterministic_2d_movement() -> None:
	board = np.zeros((10, 10), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")

	blue.navigation_algorithm = "a_star"
	blue.allow_diagonal_navigation = False
	blue.mobility = 1

	start = blue.pos
	blue.move()

	assert start == (3, 1)
	assert blue.pos == (3, 2)


def test_stage_i_board_loader_supports_csv_and_whitespace_maps(tmp_path) -> None:
	board = np.array(
		[
			[0, 1, 0],
			[0, 0, 1],
			[1, 0, 0],
		],
		dtype=int,
	)

	csv_path = tmp_path / "board.csv"
	txt_path = tmp_path / "board.txt"
	np.savetxt(csv_path, board, fmt="%d", delimiter=",")
	np.savetxt(txt_path, board, fmt="%d")

	loaded_csv = simulation_main.load_board(str(csv_path))
	loaded_txt = simulation_main.load_board(str(txt_path))

	assert np.array_equal(loaded_csv, board)
	assert np.array_equal(loaded_txt, board)


def test_stage_i_example_large_map_can_spawn_multiple_units_per_team() -> None:
	board_path = Path(__file__).resolve().parents[1] / "data" / "example_large_map.csv"
	board = simulation_main.load_board(str(board_path))
	model = BattlefieldModel(
		board,
		blue_spawn_points=[(1, 1), (2, 1), (3, 1)],
		red_spawn_points=[(12, 10), (13, 10), (14, 10)],
	)

	blue_agents = [
		agent
		for agent in model.agents
		if isinstance(agent, CombatAgent) and agent.team == "Blue"
	]
	red_agents = [
		agent
		for agent in model.agents
		if isinstance(agent, CombatAgent) and agent.team == "Red"
	]

	assert board.shape == (12, 16)
	assert len(blue_agents) == 3
	assert len(red_agents) == 3
	assert {agent.pos for agent in blue_agents} == {(1, 1), (2, 1), (3, 1)}
	assert {agent.pos for agent in red_agents} == {(12, 10), (13, 10), (14, 10)}


def test_stage_ii_pathfinding_accounts_for_terrain_topography_costs() -> None:
	terrain = np.zeros((5, 5), dtype=int)
	terrain[2, 1] = 6
	terrain[2, 2] = 6
	terrain[2, 3] = 6

	path = find_path(
		start=(0, 2),
		goal=(4, 2),
		terrain=terrain,
		algorithm="dijkstra",
		allow_diagonal=False,
	)

	assert path[0] == (0, 2)
	assert path[-1] == (4, 2)
	assert (1, 2) not in path
	assert (2, 2) not in path
	assert (3, 2) not in path


def test_stage_ii_detection_respects_los_and_vision_cone() -> None:
	board = np.zeros((8, 8), dtype=int)
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

	assert red not in blue.get_visible_enemies()

	model.terrain[3, 2] = 0
	assert red in blue.get_visible_enemies()


def test_stage_ii_combat_model_applies_damage_and_elimination() -> None:
	board = np.zeros((6, 6), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (2, 2))
	model.grid.move_agent(red, (2, 2))

	blue.attack_range = 4.0
	blue.accuracy = 10.0
	blue.firepower = 50
	red.hp = 10

	hit = blue.attack(red)

	assert hit is True
	assert red.hp == 0
	assert red not in list(model.agents)
	assert model.eliminated_by_team["Red"] == 1
	assert model.kills_by_team["Blue"] == 1


def test_stage_ii_tactical_unit_classes_define_unique_combat_profiles() -> None:
	board = np.zeros((8, 8), dtype=int)
	model = BattlefieldModel(board)

	infantry = InfantrySquad(model, team="Blue")
	recon = ReconSquad(model, team="Blue")
	mechanized = MechanizedInfantry(model, team="Blue")
	tank = MainBattleTank(model, team="Blue")

	units = [infantry, recon, mechanized, tank]
	assert all(isinstance(unit, CombatAgent) for unit in units)

	assert recon.observation_range > infantry.observation_range
	assert recon.detection_threshold < infantry.detection_threshold
	assert mechanized.mobility > infantry.mobility
	assert mechanized.firepower > infantry.firepower
	assert tank.firepower > mechanized.firepower
	assert tank.armor > mechanized.armor


def test_stage_ii_closed_decision_loop_navigation_detection_and_combat() -> None:
	board = np.zeros((10, 10), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	for agent in (blue, red):
		agent.attack_range = 4.0
		agent.accuracy = 10.0

	blue_hp_before = blue.hp
	red_hp_before = red.hp

	model.step()

	total_shots = model.shots_by_team["Blue"] + model.shots_by_team["Red"]
	total_hits = model.hits_by_team["Blue"] + model.hits_by_team["Red"]

	assert total_shots >= 1
	assert total_hits >= 1
	assert blue.hp < blue_hp_before or red.hp < red_hp_before


def test_stage_ii_ai_patrol_logic_moves_agent_between_patrol_points() -> None:
	board = np.zeros((10, 10), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(red, (9, 9))
	blue.observation_range = 2
	blue.set_patrol_route([(3, 1), (4, 1)])

	blue.step()

	assert blue.ai_state == "patrol"
	assert blue.pos == (4, 1)


def test_stage_ii_ai_retreat_logic_increases_distance_from_threat() -> None:
	board = np.zeros((10, 10), dtype=int)
	model = BattlefieldModel(board)
	blue = _agent_by_team(model, "Blue")
	red = _agent_by_team(model, "Red")

	model.grid.move_agent(blue, (3, 3))
	model.grid.move_agent(red, (3, 4))

	blue.hp = 20
	blue.max_hp = 100
	blue.retreat_health_ratio = 0.3
	blue.observation_range = 6
	blue.view_angle_deg = 180.0
	blue.facing_direction = (0, 1)

	distance_before = find_path(blue.pos, red.pos, board, algorithm="a_star")
	blue.step()
	distance_after = find_path(blue.pos, red.pos, board, algorithm="a_star")

	assert blue.ai_state == "retreat"
	assert len(distance_after) >= len(distance_before)

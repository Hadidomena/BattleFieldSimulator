import mesa
import numpy as np

from simulation.agent import CombatAgent
from simulation.telemetry import TelemetryCollector
from simulation.utils import is_position_in_bounds


class BattlefieldModel(mesa.Model):
	def __init__(
		self,
		board: np.ndarray,
		blue_unit_class: type[CombatAgent] = CombatAgent,
		red_unit_class: type[CombatAgent] = CombatAgent,
		blue_unit_kwargs: dict | None = None,
		red_unit_kwargs: dict | None = None,
		blue_spawn_points: list[tuple[int, int]] | None = None,
		red_spawn_points: list[tuple[int, int]] | None = None,
		obstacle_max_hp: float = 50.0,
		seed: int | None = None,
		spawn_agents: bool = True,
	) -> None:
		super().__init__(rng=seed)
		self.width = board.shape[1]
		self.height = board.shape[0]
		self.terrain = board.copy()
		self.blue_unit_class = blue_unit_class
		self.red_unit_class = red_unit_class
		self.blue_unit_kwargs = blue_unit_kwargs or {}
		self.red_unit_kwargs = red_unit_kwargs or {}
		self.blue_spawn_points = blue_spawn_points
		self.red_spawn_points = red_spawn_points
		self.spawn_agents = spawn_agents
		self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
		self.eliminated_by_team = {"Blue": 0, "Red": 0}
		self.kills_by_team = {"Blue": 0, "Red": 0}
		self.shots_by_team = {"Blue": 0, "Red": 0}
		self.hits_by_team = {"Blue": 0, "Red": 0}
		self.damage_by_team = {"Blue": 0, "Red": 0}
		self.obstacle_max_hp = obstacle_max_hp
		self.obstacle_hp = np.zeros_like(board, dtype=float)
		self.obstacle_hp[board == 1] = obstacle_max_hp
		self.destroyed_obstacles = 0
		self.datacollector = mesa.DataCollector(
			model_reporters={
				"Alive_Blue": lambda m: m.alive_count("Blue"),
				"Alive_Red": lambda m: m.alive_count("Red"),
				"Eliminated_Blue": lambda m: m.eliminated_by_team["Blue"],
				"Eliminated_Red": lambda m: m.eliminated_by_team["Red"],
				"Kills_Blue": lambda m: m.kills_by_team["Blue"],
				"Kills_Red": lambda m: m.kills_by_team["Red"],
				"Damage_Blue": lambda m: m.damage_by_team["Blue"],
				"Damage_Red": lambda m: m.damage_by_team["Red"],
				"Destroyed_Obstacles": lambda m: m.destroyed_obstacles,
			},
			agent_reporters={"HP": "hp"},
		)
		self.telemetry = TelemetryCollector()

		self._init_board(board)
		self.telemetry.record_step(self)
		self.running = True

	def apply_obstacle_damage(self, pos: tuple[int, int], damage: float) -> None:
		x, y = pos
		for dx in (-1, 0, 1):
			for dy in (-1, 0, 1):
				nx, ny = x + dx, y + dy
				if not (0 <= ny < self.height and 0 <= nx < self.width):
					continue
				if self.terrain[ny, nx] != 1:
					continue
				if self.obstacle_hp[ny, nx] <= 0:
					continue

				splash = damage * 0.5
				if dx == 0 and dy == 0:
					splash = damage

				self.obstacle_hp[ny, nx] -= splash
				if self.obstacle_hp[ny, nx] <= 0:
					self.terrain[ny, nx] = 0
					self.obstacle_hp[ny, nx] = 0
					self.destroyed_obstacles += 1

	def _validate_spawn_points(
		self, positions: list[tuple[int, int]], team: str
	) -> None:
		for position in positions:
			x, y = position
			if not is_position_in_bounds(position, self.terrain):
				raise ValueError(
					f"{team} spawn point {position} is out of bounds for a "
					f"{self.width}x{self.height} map"
				)
			if self.terrain[y, x] == 1:
				raise ValueError(
					f"{team} spawn point {position} is on an obstacle "
					f"(terrain == 1) and cannot be walked on"
				)

	def _init_board(self, board: np.ndarray) -> None:
		default_blue_spawn = [(3, 1)]
		default_red_spawn = [(3, 5)]
		blue_positions = self.blue_spawn_points or default_blue_spawn
		red_positions = self.red_spawn_points or default_red_spawn

		self._validate_spawn_points(blue_positions, "Blue")
		self._validate_spawn_points(red_positions, "Red")

		if not self.spawn_agents:
			return

		for spawn_position in blue_positions:
			blue_agent = self.blue_unit_class(
				self,
				team="Blue",
				**self.blue_unit_kwargs,
			)
			self.grid.place_agent(blue_agent, spawn_position)

		for spawn_position in red_positions:
			red_agent = self.red_unit_class(
				self,
				team="Red",
				**self.red_unit_kwargs,
			)
			self.grid.place_agent(red_agent, spawn_position)

	def agents_of_team(self, team: str, alive_only: bool = True) -> list[CombatAgent]:
		return [
			agent
			for agent in self.agents
			if isinstance(agent, CombatAgent)
			and agent.team == team
			and (not alive_only or agent.hp > 0)
		]

	def alive_count(self, team: str) -> int:
		return len(self.agents_of_team(team))

	def broadcast_sighting(self, team: str, position: tuple[int, int]) -> None:
		for agent in self.agents_of_team(team):
			agent.last_known_enemy_pos = position

	def is_battle_over(self) -> bool:
		return self.alive_count("Blue") == 0 or self.alive_count("Red") == 0

	def record_attack(
		self,
		team: str,
		hit: bool,
		damage: int,
		attacker_id: int = 0,
		defender_id: int = 0,
		defender_team: str = "",
	) -> None:
		if team not in self.shots_by_team:
			return

		self.shots_by_team[team] += 1
		if hit:
			self.hits_by_team[team] += 1
			self.damage_by_team[team] += max(0, int(damage))

		self.telemetry.record_shot(
			step=self.steps,
			attacker_id=attacker_id,
			attacker_team=team,
			defender_id=defender_id,
			defender_team=defender_team,
			hit=hit,
			damage=damage,
		)

	def register_elimination(
		self,
		eliminated_team: str,
		killer_team: str | None,
		eliminated_id: int = 0,
		killer_id: int | None = None,
	) -> None:
		if eliminated_team in self.eliminated_by_team:
			self.eliminated_by_team[eliminated_team] += 1

		if killer_team in self.kills_by_team:
			self.kills_by_team[killer_team] += 1

		self.telemetry.record_elimination(
			step=self.steps,
			eliminated_id=eliminated_id,
			eliminated_team=eliminated_team,
			killer_id=killer_id,
			killer_team=killer_team,
		)

	def step(self) -> None:
		self.agents.shuffle_do("step")
		self.datacollector.collect(self)
		self.telemetry.record_step(self)
		self.running = not self.is_battle_over()

import mesa
import numpy as np

from simulation.agent import CombatAgent


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
	) -> None:
		super().__init__()
		self.width = board.shape[1]
		self.height = board.shape[0]
		self.terrain = board
		self.blue_unit_class = blue_unit_class
		self.red_unit_class = red_unit_class
		self.blue_unit_kwargs = blue_unit_kwargs or {}
		self.red_unit_kwargs = red_unit_kwargs or {}
		self.blue_spawn_points = blue_spawn_points
		self.red_spawn_points = red_spawn_points
		self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
		self.eliminated_by_team = {"Blue": 0, "Red": 0}
		self.kills_by_team = {"Blue": 0, "Red": 0}
		self.shots_by_team = {"Blue": 0, "Red": 0}
		self.hits_by_team = {"Blue": 0, "Red": 0}
		self.damage_by_team = {"Blue": 0, "Red": 0}
		self.datacollector = mesa.DataCollector(
			model_reporters={
				"Alive_Blue": lambda m: sum(
					1
					for a in m.agents
					if getattr(a, "team", None) == "Blue" and getattr(a, "hp", 0) > 0
				),
				"Alive_Red": lambda m: sum(
					1
					for a in m.agents
					if getattr(a, "team", None) == "Red" and getattr(a, "hp", 0) > 0
				),
				"Eliminated_Blue": lambda m: m.eliminated_by_team["Blue"],
				"Eliminated_Red": lambda m: m.eliminated_by_team["Red"],
				"Kills_Blue": lambda m: m.kills_by_team["Blue"],
				"Kills_Red": lambda m: m.kills_by_team["Red"],
				"Damage_Blue": lambda m: m.damage_by_team["Blue"],
				"Damage_Red": lambda m: m.damage_by_team["Red"],
			},
			agent_reporters={"HP": "hp"},
		)

		self._init_board(board)
		self.running = True

	def _init_board(self, board: np.ndarray) -> None:
		"""
		TODO: Expand terrain logic
		"""
		default_blue_spawn = [(3, 1)]
		default_red_spawn = [(3, 5)]
		blue_positions = self.blue_spawn_points or default_blue_spawn
		red_positions = self.red_spawn_points or default_red_spawn

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

	def is_battle_over(self) -> bool:
		alive_blue = sum(
			1
			for a in self.agents
			if getattr(a, "team", None) == "Blue" and getattr(a, "hp", 0) > 0
		)
		alive_red = sum(
			1
			for a in self.agents
			if getattr(a, "team", None) == "Red" and getattr(a, "hp", 0) > 0
		)
		return alive_blue == 0 or alive_red == 0

	def record_attack(self, team: str, hit: bool, damage: int) -> None:
		if team not in self.shots_by_team:
			return

		self.shots_by_team[team] += 1
		if hit:
			self.hits_by_team[team] += 1
			self.damage_by_team[team] += max(0, int(damage))

	def register_elimination(
		self,
		eliminated_team: str,
		killer_team: str | None,
	) -> None:
		if eliminated_team in self.eliminated_by_team:
			self.eliminated_by_team[eliminated_team] += 1

		if killer_team in self.kills_by_team:
			self.kills_by_team[killer_team] += 1

	def step(self) -> None:
		self.datacollector.collect(self)
		self.agents.shuffle_do("step")
		self.running = not self.is_battle_over()

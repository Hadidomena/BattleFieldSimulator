import mesa
import numpy as np

from simulation.agent import CombatAgent


class BattlefieldModel(mesa.Model):
	def __init__(self, board: np.ndarray) -> None:
		super().__init__()
		self.width = board.shape[1]
		self.height = board.shape[0]
		self.terrain = board
		self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
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
			},
			agent_reporters={"HP": "hp"},
		)

		self._init_board(board)
		self.running = True

	def _init_board(self, board: np.ndarray) -> None:
		"""
		TODO: Terrain
		"""
		blue_agent = CombatAgent(self, team="Blue")
		self.grid.place_agent(blue_agent, (0, 0))
		red_agent = CombatAgent(self, team="Red")
		target_pos = (self.width - 1, self.height - 1)
		self.grid.place_agent(red_agent, target_pos)

	def step(self) -> None:
		self.datacollector.collect(self)
		self.agents.shuffle_do("step")

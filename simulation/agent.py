import mesa
import numpy as np

from simulation.utils import (
	detection_score,
	euclidean_distance,
	find_path,
	visible_cells,
)


class CombatAgent(mesa.Agent):
	def __init__(
		self,
		model: mesa.Model,
		team: str,
		hp: int = 100,
		firepower: int = 10,
		observation_range: int = 5,
		view_angle_deg: float = 120.0,
		detection_threshold: float = 0.15,
		facing_direction: tuple[int, int] = (0, 1),
		mobility: int = 2,
		navigation_algorithm: str = "a_star",
		allow_diagonal_navigation: bool = False,
	) -> None:
		super().__init__(model)
		self.team = team
		self.hp = hp
		self.firepower = firepower
		self.observation_range = observation_range
		self.view_angle_deg = view_angle_deg
		self.detection_threshold = detection_threshold
		self.facing_direction = facing_direction
		self.mobility = mobility
		self.navigation_algorithm = navigation_algorithm
		self.allow_diagonal_navigation = allow_diagonal_navigation
		self.last_detection_scores: dict[int, float] = {}
		self.current_path: list[tuple[int, int]] = []

	def step(self) -> None:
		if self.hp <= 0:
			return

		self.move()

		enemies = self.get_visible_enemies()
		enemy_ids = [e.unique_id for e in enemies]

		current_step = self.model.steps
		print(
			f"[Turn {current_step}] Agent {self.unique_id} "
			f"({self.team}) in {self.pos} ready. HP: {self.hp}. "
			f"Visible enemies: {enemy_ids}"
		)

	def get_visible_enemies(self) -> list["CombatAgent"]:
		enemies_with_score: list[tuple[CombatAgent, float]] = []
		for agent in self.model.agents:
			if (
				isinstance(agent, CombatAgent)
				and agent.team != self.team
				and agent.hp > 0
			):
				score = detection_score(
					observer_pos=self.pos,
					target_pos=agent.pos,
					terrain=self.model.terrain,
					observation_range=self.observation_range,
					facing_direction=self.facing_direction,
					view_angle_deg=self.view_angle_deg,
				)
				if score >= self.detection_threshold:
					enemies_with_score.append((agent, score))

		enemies_with_score.sort(key=lambda item: item[1], reverse=True)
		self.last_detection_scores = {
			agent.unique_id: score for agent, score in enemies_with_score
		}
		return [agent for agent, _ in enemies_with_score]

	def get_visible_cells(self) -> set[tuple[int, int]]:
		return visible_cells(
			observer_pos=self.pos,
			terrain=self.model.terrain,
			observation_range=self.observation_range,
			facing_direction=self.facing_direction,
			view_angle_deg=self.view_angle_deg,
		)

	def _get_alive_enemy_agents(self) -> list["CombatAgent"]:
		try:
			agents = list(self.model.agents)
		except TypeError:
			return []

		return [
			agent
			for agent in agents
			if (
				isinstance(agent, CombatAgent)
				and agent.team != self.team
				and agent.hp > 0
			)
		]

	def _get_navigation_target(self) -> tuple[int, int] | None:
		visible_enemies = self.get_visible_enemies()
		if visible_enemies:
			return visible_enemies[0].pos

		enemies = self._get_alive_enemy_agents()
		if not enemies:
			return None

		return min(
			enemies,
			key=lambda enemy: euclidean_distance(self.pos, enemy.pos),
		).pos

	def _move_randomly(self) -> None:
		if self.mobility <= 0:
			return

		try:
			possible_steps = list(
				self.model.grid.get_neighborhood(
					self.pos, moore=True, include_center=False
				)
			)
		except (TypeError, AttributeError):
			return

		terrain = getattr(self.model, "terrain", None)
		if isinstance(terrain, np.ndarray):
			valid_steps = [pos for pos in possible_steps if terrain[pos[1], pos[0]] != 1]
		else:
			valid_steps = possible_steps

		if valid_steps:
			old_position = self.pos
			new_position = self.random.choice(valid_steps)
			self.model.grid.move_agent(self, new_position)
			dx = new_position[0] - old_position[0]
			dy = new_position[1] - old_position[1]
			if (dx, dy) != (0, 0):
				self.facing_direction = (dx, dy)

	def move(self) -> None:
		terrain = getattr(self.model, "terrain", None)
		if not isinstance(terrain, np.ndarray):
			self._move_randomly()
			return

		if self.mobility <= 0:
			return

		target_position = self._get_navigation_target()
		if target_position is None:
			self._move_randomly()
			return

		path = find_path(
			start=self.pos,
			goal=target_position,
			terrain=terrain,
			algorithm=self.navigation_algorithm,
			allow_diagonal=self.allow_diagonal_navigation,
		)

		if len(path) <= 1:
			self.current_path = []
			self._move_randomly()
			return

		step_index = min(self.mobility, len(path) - 1)
		old_position = self.pos
		new_position = path[step_index]
		self.model.grid.move_agent(self, new_position)
		self.current_path = path

		direction_step = path[1]
		dx = direction_step[0] - old_position[0]
		dy = direction_step[1] - old_position[1]
		if (dx, dy) != (0, 0):
			self.facing_direction = (dx, dy)

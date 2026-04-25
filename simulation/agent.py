from __future__ import annotations

import mesa
import numpy as np

from simulation.utils import (
	calculate_damage,
	cover_ratio,
	detection_score,
	euclidean_distance,
	find_path,
	has_line_of_sight,
	hit_probability,
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
		attack_range: float = 3.0,
		view_angle_deg: float = 120.0,
		detection_threshold: float = 0.15,
		accuracy: float = 0.75,
		armor: float = 0.10,
		facing_direction: tuple[int, int] = (0, 1),
		mobility: int = 2,
		navigation_algorithm: str = "a_star",
		allow_diagonal_navigation: bool = False,
		retreat_health_ratio: float = 0.30,
	) -> None:
		super().__init__(model)
		self.team = team
		self.hp = hp
		self.firepower = firepower
		self.observation_range = observation_range
		self.attack_range = attack_range
		self.view_angle_deg = view_angle_deg
		self.detection_threshold = detection_threshold
		self.accuracy = accuracy
		self.armor = armor
		self.facing_direction = facing_direction
		self.mobility = mobility
		self.navigation_algorithm = navigation_algorithm
		self.allow_diagonal_navigation = allow_diagonal_navigation
		self.retreat_health_ratio = retreat_health_ratio
		self.last_detection_scores: dict[int, float] = {}
		self.current_path: list[tuple[int, int]] = []
		self.last_damage_dealt: int = 0
		self.last_damage_taken: int = 0
		self.max_hp: int = hp
		self.ai_state: str = "advance"
		self.patrol_route: list[tuple[int, int]] = []
		self.patrol_index: int = 0

	def step(self) -> None:
		if self.hp <= 0:
			return

		visible_before_move = self.get_visible_enemies()
		self.update_behavior_state(visible_before_move)

		if self.ai_state == "retreat":
			self.retreat(visible_before_move)
		elif self.ai_state == "engage":
			engaged = self.attack_closest_target(visible_before_move)
			if not engaged:
				self.move()
				visible_after_move = self.get_visible_enemies()
				self.attack_closest_target(visible_after_move)
		elif self.ai_state == "patrol":
			self.patrol_step()
			visible_after_move = self.get_visible_enemies()
			self.attack_closest_target(visible_after_move)
		else:
			self.move()
			visible_after_move = self.get_visible_enemies()
			self.attack_closest_target(visible_after_move)

		enemies = self.get_visible_enemies()
		enemy_ids = [e.unique_id for e in enemies]

		current_step = self.model.steps
		print(
			f"[Turn {current_step}] Agent {self.unique_id} "
			f"({self.team}) in {self.pos} ready. HP: {self.hp}. State: {self.ai_state}. "
			f"Visible enemies: {enemy_ids}"
		)

	def update_behavior_state(self, visible_enemies: list["CombatAgent"]) -> None:
		has_visible_enemy = len(visible_enemies) > 0
		health_ratio = self.hp / max(1, self.max_hp)

		if has_visible_enemy and health_ratio <= self.retreat_health_ratio:
			self.ai_state = "retreat"
			return
		if has_visible_enemy:
			self.ai_state = "engage"
			return
		if self.patrol_route:
			self.ai_state = "patrol"
		else:
			self.ai_state = "advance"

	def set_patrol_route(self, patrol_route: list[tuple[int, int]]) -> None:
		self.patrol_route = list(patrol_route)
		self.patrol_index = 0

	def patrol_step(self) -> None:
		if not self.patrol_route:
			self.move()
			return

		target = self.patrol_route[self.patrol_index]
		if self.pos == target:
			self.patrol_index = (self.patrol_index + 1) % len(self.patrol_route)
			target = self.patrol_route[self.patrol_index]

		self.move(target_position=target)

	def retreat(self, visible_enemies: list["CombatAgent"]) -> None:
		if not visible_enemies:
			self.move()
			return

		threat = min(
			visible_enemies,
			key=lambda enemy: euclidean_distance(self.pos, enemy.pos),
		)

		neighbors = list(
			self.model.grid.get_neighborhood(
				self.pos,
				moore=True,
				include_center=False,
			)
		)
		terrain = getattr(self.model, "terrain", None)
		if isinstance(terrain, np.ndarray):
			neighbors = [cell for cell in neighbors if terrain[cell[1], cell[0]] != 1]

		if not neighbors:
			return

		best_cell = max(
			neighbors,
			key=lambda cell: euclidean_distance(cell, threat.pos),
		)
		old_position = self.pos
		self.model.grid.move_agent(self, best_cell)
		dx = best_cell[0] - old_position[0]
		dy = best_cell[1] - old_position[1]
		if (dx, dy) != (0, 0):
			self.facing_direction = (dx, dy)

	def _is_in_attack_range(self, target: "CombatAgent") -> bool:
		distance = euclidean_distance(self.pos, target.pos)
		return distance <= self.attack_range

	def _pick_attack_target(
		self,
		visible_enemies: list["CombatAgent"],
	) -> "CombatAgent" | None:
		attackable = [
			enemy for enemy in visible_enemies if self._is_in_attack_range(enemy)
		]
		if not attackable:
			return None
		return min(attackable, key=lambda enemy: euclidean_distance(self.pos, enemy.pos))

	def attack_closest_target(self, visible_enemies: list["CombatAgent"]) -> bool:
		target = self._pick_attack_target(visible_enemies)
		if target is None:
			return False
		return self.attack(target)

	def attack(self, target: "CombatAgent") -> bool:
		if self.hp <= 0 or target.hp <= 0:
			return False
		if target.pos is None:
			return False

		distance = euclidean_distance(self.pos, target.pos)
		if distance > self.attack_range:
			return False
		if not has_line_of_sight(self.pos, target.pos, self.model.terrain):
			return False

		cover = cover_ratio(target.pos, self.model.terrain)
		hit_chance = hit_probability(
			distance=distance,
			attack_range=self.attack_range,
			base_accuracy=self.accuracy,
			cover=cover,
		)

		roll = self.random.random()
		if roll > hit_chance:
			if hasattr(self.model, "record_attack"):
				self.model.record_attack(self.team, hit=False, damage=0)
			self.last_damage_dealt = 0
			return False

		damage = calculate_damage(
			firepower=self.firepower,
			distance=distance,
			attack_range=self.attack_range,
			cover=cover,
			armor=target.armor,
		)

		target.receive_damage(damage, attacker=self)
		if hasattr(self.model, "record_attack"):
			self.model.record_attack(self.team, hit=True, damage=damage)
		self.last_damage_dealt = damage
		return True

	def receive_damage(self, amount: int, attacker: "CombatAgent" | None = None) -> None:
		if self.hp <= 0:
			return

		damage = max(0, int(amount))
		self.last_damage_taken = damage
		self.hp = max(0, self.hp - damage)

		if self.hp == 0:
			killer_team = attacker.team if attacker else None
			self.eliminate(killer_team)

	def eliminate(self, killer_team: str | None = None) -> None:
		self.hp = 0
		if hasattr(self.model, "register_elimination"):
			self.model.register_elimination(
				eliminated_team=self.team,
				killer_team=killer_team,
			)

		if self.pos is not None and hasattr(self.model, "grid"):
			try:
				self.model.grid.remove_agent(self)
			except Exception:
				pass

		try:
			self.remove()
		except Exception:
			pass

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
		if self.patrol_route:
			return self.patrol_route[self.patrol_index]

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

	def move(self, target_position: tuple[int, int] | None = None) -> None:
		terrain = getattr(self.model, "terrain", None)
		if not isinstance(terrain, np.ndarray):
			self._move_randomly()
			return

		if self.mobility <= 0:
			return

		target_position = target_position or self._get_navigation_target()
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


class InfantrySquad(CombatAgent):
	"""
	Basic squad
	"""

	def __init__(
		self,
		model: mesa.Model,
		team: str,
		hp: int = 100,
		firepower: int = 10,
		observation_range: int = 6,
		attack_range: float = 4.0,
		view_angle_deg: float = 120.0,
		detection_threshold: float = 0.15,
		accuracy: float = 0.70,
		armor: float = 0.10,
		facing_direction: tuple[int, int] = (0, 1),
		mobility: int = 2,
		navigation_algorithm: str = "a_star",
		allow_diagonal_navigation: bool = False,
	) -> None:
		super().__init__(
			model,
			team,
			hp,
			firepower,
			observation_range,
			attack_range,
			view_angle_deg,
			detection_threshold,
			accuracy,
			armor,
			facing_direction,
			mobility,
			navigation_algorithm,
			allow_diagonal_navigation,
		)


class ReconSquad(CombatAgent):
	"""
	Recon Squad which is worse in sustained contact,
	but faster
	"""

	def __init__(
		self,
		model: mesa.Model,
		team: str,
		hp: int = 80,
		firepower: int = 8,
		observation_range: int = 12,
		attack_range: float = 5.0,
		view_angle_deg: float = 180.0,
		detection_threshold: float = 0.08,
		accuracy: float = 0.75,
		armor: float = 0.05,
		facing_direction: tuple[int, int] = (0, 1),
		mobility: int = 3,
		navigation_algorithm: str = "a_star",
		allow_diagonal_navigation: bool = False,
	) -> None:
		super().__init__(
			model,
			team,
			hp,
			firepower,
			observation_range,
			attack_range,
			view_angle_deg,
			detection_threshold,
			accuracy,
			armor,
			facing_direction,
			mobility,
			navigation_algorithm,
			allow_diagonal_navigation,
		)


class MechanizedInfantry(CombatAgent):
	"""
	more mobile and durable than InfantrySquad
	TODO: maybe in futre implement more of weaknesses
	"""

	def __init__(
		self,
		model: mesa.Model,
		team: str,
		hp: int = 200,
		firepower: int = 20,
		observation_range: int = 8,
		attack_range: float = 8.0,
		view_angle_deg: float = 120.0,
		detection_threshold: float = 0.15,
		accuracy: float = 0.80,
		armor: float = 0.30,
		facing_direction: tuple[int, int] = (0, 1),
		mobility: int = 4,
		navigation_algorithm: str = "a_star",
		allow_diagonal_navigation: bool = False,
	) -> None:
		super().__init__(
			model,
			team,
			hp,
			firepower,
			observation_range,
			attack_range,
			view_angle_deg,
			detection_threshold,
			accuracy,
			armor,
			facing_direction,
			mobility,
			navigation_algorithm,
			allow_diagonal_navigation,
		)


class MainBattleTank(CombatAgent):
	"""
	Tank, more durable but slower than MechanizedInfantry
	"""

	def __init__(
		self,
		model: mesa.Model,
		team: str,
		hp: int = 400,
		firepower: int = 40,
		observation_range: int = 10,
		attack_range: float = 12.0,
		view_angle_deg: float = 90.0,
		detection_threshold: float = 0.10,
		accuracy: float = 0.85,
		armor: float = 0.60,
		facing_direction: tuple[int, int] = (0, 1),
		mobility: int = 3,
		navigation_algorithm: str = "a_star",
		allow_diagonal_navigation: bool = False,
	) -> None:
		super().__init__(
			model,
			team,
			hp,
			firepower,
			observation_range,
			attack_range,
			view_angle_deg,
			detection_threshold,
			accuracy,
			armor,
			facing_direction,
			mobility,
			navigation_algorithm,
			allow_diagonal_navigation,
		)

from __future__ import annotations

import math

import mesa
import numpy as np

from simulation.utils import (
	calculate_damage,
	cover_ratio,
	detection_score,
	directional_cover_ratio,
	euclidean_distance,
	find_path,
	has_line_of_sight,
	hit_probability,
	is_in_vision_cone,
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
		cover_multiplier: float = 1.0,
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
		self.cover_multiplier = cover_multiplier
		self.last_detection_scores: dict[int, float] = {}
		self.current_path: list[tuple[int, int]] = []
		self.last_damage_dealt: int = 0
		self.last_damage_taken: int = 0
		self._last_attacker_id: int | None = None
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
			if not self._covering_fire(visible_before_move):
				self._covering_fire_suppress(visible_before_move)
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

	def _get_nearby_allies(
		self, pos: tuple[int, int], radius: float
	) -> list["CombatAgent"]:
		allies: list["CombatAgent"] = []
		for agent in self.model.agents:
			if (
				isinstance(agent, CombatAgent)
				and agent.team == self.team
				and agent.hp > 0
				and agent.unique_id != self.unique_id
				and euclidean_distance(pos, agent.pos) <= radius
			):
				allies.append(agent)
		return allies

	def _get_ally_positions(self) -> list[tuple[int, int]]:
		positions: list[tuple[int, int]] = []
		for agent in self.model.agents:
			if (
				isinstance(agent, CombatAgent)
				and agent.team == self.team
				and agent.hp > 0
				and agent.unique_id != self.unique_id
				and agent.pos is not None
			):
				positions.append(agent.pos)
		return positions

	def _covering_fire(self, visible_enemies: list["CombatAgent"]) -> bool:
		target = self._pick_attack_target(visible_enemies)
		if target is None:
			return False
		return self.attack(target, accuracy_override=self.accuracy * 0.5)

	def _covering_fire_suppress(self, visible_enemies: list["CombatAgent"]) -> bool:
		target = self._pick_attack_target(visible_enemies)
		if target is None:
			return False
		return self.attack(target, accuracy_override=self.accuracy * 0.35)

	def _score_retreat_cell(
		self,
		candidate: tuple[int, int],
		enemy_positions: list[tuple[int, int]],
		ally_positions: list[tuple[int, int]],
		group_centroid: tuple[float, float] | None,
		max_map_dist: float,
	) -> float:
		terrain = getattr(self.model, "terrain", None)
		if terrain is None:
			return float("-inf")

		dist_from_enemies = min(
			euclidean_distance(candidate, ep) for ep in enemy_positions
		)

		los_blocked = 0
		for ep in enemy_positions:
			if not has_line_of_sight(candidate, ep, terrain):
				los_blocked += 1

		cover = cover_ratio(candidate, terrain)

		nearby_allies_at_cell = 0
		for ap in ally_positions:
			if euclidean_distance(candidate, ap) <= self.attack_range:
				nearby_allies_at_cell += 1

		group_cohesion = 0.0
		if group_centroid is not None:
			dg = euclidean_distance(candidate, group_centroid)
			group_cohesion = max(0.0, 1.0 - dg / self.observation_range)

		normalized_dist = dist_from_enemies / max_map_dist

		return (
			normalized_dist * 10.0
			+ los_blocked * 5.0
			+ cover * 8.0
			+ nearby_allies_at_cell * 3.0
			+ group_cohesion * 5.0
		)

	def retreat(self, visible_enemies: list["CombatAgent"]) -> None:  # noqa: C901
		terrain = getattr(self.model, "terrain", None)
		if not isinstance(terrain, np.ndarray):
			self.move()
			self._covering_fire(self.get_visible_enemies())
			return

		enemy_positions = [e.pos for e in visible_enemies if e.pos is not None]
		if not enemy_positions:
			self.move()
			return

		ally_positions = self._get_ally_positions()

		nearby_allies = self._get_nearby_allies(self.pos, self.attack_range * 2)
		if len(visible_enemies) > len(nearby_allies) + 1:
			self.ai_state = "engage"
			self.attack_closest_target(visible_enemies)
			return

		retreat_group: list["CombatAgent"] = []
		for agent in self.model.agents:
			if (
				isinstance(agent, CombatAgent)
				and agent.team == self.team
				and agent.hp > 0
				and agent.unique_id != self.unique_id
				and agent.ai_state == "retreat"
				and agent.pos is not None
				and euclidean_distance(self.pos, agent.pos) <= self.observation_range
			):
				retreat_group.append(agent)

		group_centroid: tuple[float, float] | None = None
		if retreat_group:
			cx = sum(a.pos[0] for a in retreat_group) / len(retreat_group)
			cy = sum(a.pos[1] for a in retreat_group) / len(retreat_group)
			group_centroid = (cx, cy)

		best_cell: tuple[int, int] | None = None
		best_score = float("-inf")

		max_map_dist = (
			math.hypot(
				getattr(self.model, "width", 20), getattr(self.model, "height", 20)
			)
			or 1.0
		)

		search_radius = max(4, self.mobility + 2)
		for dx in range(-search_radius, search_radius + 1):
			for dy in range(-search_radius, search_radius + 1):
				if dx == 0 and dy == 0:
					continue
				candidate = (self.pos[0] + dx, self.pos[1] + dy)
				if not _is_position_valid(candidate, terrain):
					continue

				score = self._score_retreat_cell(
					candidate,
					enemy_positions,
					ally_positions,
					group_centroid,
					max_map_dist,
				)

				if score > best_score:
					best_score = score
					best_cell = candidate

		if best_cell is None:
			self.move()
			self._covering_fire(self.get_visible_enemies())
			return

		path = find_path(
			start=self.pos,
			goal=best_cell,
			terrain=terrain,
			algorithm=self.navigation_algorithm,
			allow_diagonal=self.allow_diagonal_navigation,
		)

		if len(path) <= 1:
			self._move_randomly()
			self._covering_fire(self.get_visible_enemies())
			return

		step_index = min(self.mobility, len(path) - 1)
		old_position = self.pos
		new_position = path[step_index]
		self.model.grid.move_agent(self, new_position)

		if len(path) >= 2:
			dx = path[1][0] - old_position[0]
			dy = path[1][1] - old_position[1]
			if (dx, dy) != (0, 0):
				self.facing_direction = (dx, dy)

		self._covering_fire(self.get_visible_enemies())

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

	def attack(
		self,
		target: "CombatAgent",
		accuracy_override: float | None = None,
	) -> bool:
		if self.hp <= 0 or target.hp <= 0:
			return False
		if target.pos is None:
			return False

		distance = euclidean_distance(self.pos, target.pos)
		if distance > self.attack_range:
			return False
		if not has_line_of_sight(self.pos, target.pos, self.model.terrain):
			return False

		cover = directional_cover_ratio(target.pos, self.pos, self.model.terrain)
		cover *= getattr(target, "cover_multiplier", 1.0)

		attacker_cover = directional_cover_ratio(
			self.pos, target.pos, self.model.terrain
		)
		attacker_cover *= getattr(self, "cover_multiplier", 1.0)

		base_acc = accuracy_override if accuracy_override is not None else self.accuracy
		hit_chance = hit_probability(
			distance=distance,
			attack_range=self.attack_range,
			base_accuracy=base_acc,
			cover=cover,
			attacker_cover=attacker_cover,
		)

		roll = self.random.random()
		if roll > hit_chance:
			if hasattr(self.model, "record_attack"):
				self.model.record_attack(
					self.team,
					hit=False,
					damage=0,
					attacker_id=self.unique_id,
					defender_id=target.unique_id,
					defender_team=target.team,
				)
			self.last_damage_dealt = 0
			return False

		flanking_multiplier = 1.0
		if not is_in_vision_cone(
			target.pos,
			self.pos,
			target.facing_direction,
			target.view_angle_deg,
		):
			flanking_multiplier = 1.5

		damage = calculate_damage(
			firepower=self.firepower,
			distance=distance,
			attack_range=self.attack_range,
			cover=cover,
			armor=target.armor,
			flanking_multiplier=flanking_multiplier,
		)

		if hasattr(self.model, "apply_obstacle_damage") and target.pos is not None:
			self.model.apply_obstacle_damage(target.pos, damage, self.team)
		target.receive_damage(damage, attacker=self)
		if hasattr(self.model, "record_attack"):
			self.model.record_attack(
				self.team,
				hit=True,
				damage=damage,
				attacker_id=self.unique_id,
				defender_id=target.unique_id,
				defender_team=target.team,
			)
		self.last_damage_dealt = damage
		return True

	def receive_damage(self, amount: int, attacker: "CombatAgent" | None = None) -> None:
		if self.hp <= 0:
			return

		damage = max(0, int(amount))
		self.last_damage_taken = damage
		self.hp = max(0, self.hp - damage)

		if attacker is not None:
			self._last_attacker_id = attacker.unique_id

		if self.hp == 0:
			killer_team = attacker.team if attacker else None
			self.eliminate(killer_team)

	def eliminate(self, killer_team: str | None = None) -> None:
		self.hp = 0
		if hasattr(self.model, "register_elimination"):
			self.model.register_elimination(
				eliminated_team=self.team,
				killer_team=killer_team,
				eliminated_id=self.unique_id,
				killer_id=self._last_attacker_id,
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
		retreat_health_ratio: float = 0.30,
		cover_multiplier: float = 1.0,
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
			retreat_health_ratio=retreat_health_ratio,
			cover_multiplier=cover_multiplier,
		)


class ReconSquad(CombatAgent):
	"""
	Recon Squad which is worse in sustained contact,
	but faster. Uses cover and LoS breaks to avoid
	combat longer.
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
		retreat_health_ratio: float = 0.40,
		cover_multiplier: float = 1.0,
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
			retreat_health_ratio=retreat_health_ratio,
			cover_multiplier=cover_multiplier,
		)

	def _score_retreat_cell(
		self,
		candidate: tuple[int, int],
		enemy_positions: list[tuple[int, int]],
		ally_positions: list[tuple[int, int]],
		group_centroid: tuple[float, float] | None,
		max_map_dist: float,
	) -> float:
		terrain = getattr(self.model, "terrain", None)
		if terrain is None:
			return float("-inf")

		dist_from_enemies = min(
			euclidean_distance(candidate, ep) for ep in enemy_positions
		)

		los_blocked = 0
		for ep in enemy_positions:
			if not has_line_of_sight(candidate, ep, terrain):
				los_blocked += 1

		cover = cover_ratio(candidate, terrain)

		nearby_allies_at_cell = 0
		for ap in ally_positions:
			if euclidean_distance(candidate, ap) <= self.attack_range:
				nearby_allies_at_cell += 1

		group_cohesion = 0.0
		if group_centroid is not None:
			dg = euclidean_distance(candidate, group_centroid)
			group_cohesion = max(0.0, 1.0 - dg / self.observation_range)

		normalized_dist = dist_from_enemies / max_map_dist

		return (
			normalized_dist * 10.0
			+ los_blocked * 8.0
			+ cover * 12.0
			+ nearby_allies_at_cell * 1.0
			+ group_cohesion * 3.0
		)

	def _find_cover_waypoint(
		self, target_pos: tuple[int, int]
	) -> tuple[int, int] | None:
		terrain = getattr(self.model, "terrain", None)
		if terrain is None:
			return None

		search_radius = max(3, self.mobility + 1)
		best_pos: tuple[int, int] | None = None
		best_score = float("-inf")
		current_dist = euclidean_distance(self.pos, target_pos)

		for dx in range(-search_radius, search_radius + 1):
			for dy in range(-search_radius, search_radius + 1):
				candidate = (self.pos[0] + dx, self.pos[1] + dy)
				if not _is_position_valid(candidate, terrain):
					continue
				if candidate == self.pos:
					continue

				cover = cover_ratio(candidate, terrain)
				cand_dist = euclidean_distance(candidate, target_pos)

				if cand_dist >= current_dist and cover <= 0:
					continue

				progress = max(0, current_dist - cand_dist) / max(1, current_dist)
				score = cover * 12.0 + progress * 5.0

				if score > best_score:
					best_score = score
					best_pos = candidate

		return best_pos

	def move(self, target_position: tuple[int, int] | None = None) -> None:
		if target_position is not None:
			super().move(target_position)
			return

		if self.ai_state in ("advance", "engage"):
			navigation_target = self._get_navigation_target()
			if navigation_target is not None:
				cover_pos = self._find_cover_waypoint(navigation_target)
				if cover_pos is not None:
					super().move(cover_pos)
					return

		super().move(target_position)

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
		cover_multiplier: float = 0.7,
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
			cover_multiplier=cover_multiplier,
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
		observation_range: int = 5,
		attack_range: float = 12.0,
		view_angle_deg: float = 90.0,
		detection_threshold: float = 0.10,
		accuracy: float = 0.85,
		armor: float = 0.60,
		facing_direction: tuple[int, int] = (0, 1),
		mobility: int = 3,
		navigation_algorithm: str = "a_star",
		allow_diagonal_navigation: bool = False,
		retreat_health_ratio: float = 0.30,
		cover_multiplier: float = 0.3,
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
			retreat_health_ratio=retreat_health_ratio,
			cover_multiplier=cover_multiplier,
		)


def _is_position_valid(pos: tuple[int, int], terrain: np.ndarray) -> bool:
	x, y = pos
	if x < 0 or y < 0:
		return False
	if y >= terrain.shape[0] or x >= terrain.shape[1]:
		return False
	return int(terrain[y, x]) != 1

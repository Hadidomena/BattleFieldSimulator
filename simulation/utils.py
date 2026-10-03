import heapq
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

type Position = tuple[int, int]


def load_board(path: str | Path, base_dir: str | Path | None = None) -> np.ndarray:
	if base_dir is not None:
		path = Path(base_dir) / path
	board_path = Path(path)
	if not board_path.exists():
		raise FileNotFoundError(f"Map file not found: {board_path}")
	try:
		return np.loadtxt(board_path, delimiter=",", dtype=int)
	except Exception:
		return np.loadtxt(board_path, dtype=int)


def run_model(
	model: Any, steps: int, on_step: Callable[[Any], None] | None = None
) -> None:
	for _ in range(steps):
		if not model.running:
			break
		model.step()
		if on_step is not None:
			on_step(model)


def is_position_in_bounds(pos: Position, terrain: np.ndarray) -> bool:
	x, y = pos
	return 0 <= y < terrain.shape[0] and 0 <= x < terrain.shape[1]


def euclidean_distance(pos1: Position, pos2: Position) -> float:
	dx = pos1[0] - pos2[0]
	dy = pos1[1] - pos2[1]
	return math.hypot(dx, dy)


def terrain_movement_cost(tile_value: int) -> float:
	if tile_value == 1:
		return math.inf
	if tile_value <= 0:
		return 1.0
	return float(tile_value)


def is_walkable(pos: Position, terrain: np.ndarray) -> bool:
	if not is_position_in_bounds(pos, terrain):
		return False
	x, y = pos
	return int(terrain[y, x]) != 1


def get_walkable_neighbors(
	pos: Position,
	terrain: np.ndarray,
	allow_diagonal: bool = False,
) -> list[Position]:
	x, y = pos
	directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
	if allow_diagonal:
		directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

	neighbors: list[Position] = []
	for dx, dy in directions:
		candidate = (x + dx, y + dy)
		if not is_walkable(candidate, terrain):
			continue

		if allow_diagonal and dx != 0 and dy != 0:
			horizontal = (x + dx, y)
			vertical = (x, y + dy)
			if not is_walkable(horizontal, terrain):
				continue
			if not is_walkable(vertical, terrain):
				continue

		neighbors.append(candidate)

	return neighbors


def iter_walkable_cells(
	center: Position,
	radius: int,
	terrain: np.ndarray,
) -> Iterator[Position]:
	cx, cy = center
	for dx in range(-radius, radius + 1):
		for dy in range(-radius, radius + 1):
			if dx == 0 and dy == 0:
				continue
			candidate = (cx + dx, cy + dy)
			if is_walkable(candidate, terrain):
				yield candidate


def movement_step_cost(
	current: Position,
	target: Position,
	terrain: np.ndarray,
) -> float:
	tile_cost = terrain_movement_cost(int(terrain[target[1], target[0]]))
	if math.isinf(tile_cost):
		return math.inf

	is_diagonal = current[0] != target[0] and current[1] != target[1]
	if is_diagonal:
		return tile_cost * math.sqrt(2)
	return tile_cost


def _minimum_traversal_cost(terrain: np.ndarray) -> float:
	walkable_values = terrain[terrain != 1]
	if walkable_values.size == 0:
		return 1.0
	if np.any(walkable_values <= 0):
		return 1.0
	return float(np.min(walkable_values))


def heuristic_cost(
	start: Position,
	goal: Position,
	min_tile_cost: float,
	allow_diagonal: bool,
) -> float:
	dx = abs(start[0] - goal[0])
	dy = abs(start[1] - goal[1])

	if allow_diagonal:
		d = min_tile_cost
		d2 = min_tile_cost * math.sqrt(2)
		return d * (dx + dy) + (d2 - 2 * d) * min(dx, dy)

	return min_tile_cost * (dx + dy)


def reconstruct_path(
	came_from: dict[Position, Position],
	current: Position,
) -> list[Position]:
	path = [current]
	while current in came_from:
		current = came_from[current]
		path.append(current)
	path.reverse()
	return path


def a_star_path(
	start: Position,
	goal: Position,
	terrain: np.ndarray,
	allow_diagonal: bool = False,
) -> list[Position]:
	if not is_walkable(start, terrain) or not is_walkable(goal, terrain):
		return []
	if start == goal:
		return [start]

	min_tile_cost = _minimum_traversal_cost(terrain)
	open_heap: list[tuple[float, Position]] = []
	heapq.heappush(
		open_heap,
		(
			heuristic_cost(
				start=start,
				goal=goal,
				min_tile_cost=min_tile_cost,
				allow_diagonal=allow_diagonal,
			),
			start,
		),
	)

	came_from: dict[Position, Position] = {}
	g_score: dict[Position, float] = {start: 0.0}
	closed: set[Position] = set()

	while open_heap:
		_, current = heapq.heappop(open_heap)
		if current in closed:
			continue
		if current == goal:
			return reconstruct_path(came_from, current)

		closed.add(current)
		for neighbor in get_walkable_neighbors(
			pos=current,
			terrain=terrain,
			allow_diagonal=allow_diagonal,
		):
			tentative = g_score[current] + movement_step_cost(current, neighbor, terrain)
			if tentative >= g_score.get(neighbor, math.inf):
				continue

			came_from[neighbor] = current
			g_score[neighbor] = tentative
			estimated_total = tentative + heuristic_cost(
				start=neighbor,
				goal=goal,
				min_tile_cost=min_tile_cost,
				allow_diagonal=allow_diagonal,
			)
			heapq.heappush(open_heap, (estimated_total, neighbor))

	return []


def dijkstra_path(
	start: Position,
	goal: Position,
	terrain: np.ndarray,
	allow_diagonal: bool = False,
) -> list[Position]:
	if not is_walkable(start, terrain) or not is_walkable(goal, terrain):
		return []
	if start == goal:
		return [start]

	open_heap: list[tuple[float, Position]] = [(0.0, start)]
	came_from: dict[Position, Position] = {}
	distance: dict[Position, float] = {start: 0.0}

	while open_heap:
		current_distance, current = heapq.heappop(open_heap)
		if current_distance > distance.get(current, math.inf):
			continue
		if current == goal:
			return reconstruct_path(came_from, current)

		for neighbor in get_walkable_neighbors(
			pos=current,
			terrain=terrain,
			allow_diagonal=allow_diagonal,
		):
			new_distance = current_distance + movement_step_cost(
				current, neighbor, terrain
			)
			if new_distance >= distance.get(neighbor, math.inf):
				continue

			distance[neighbor] = new_distance
			came_from[neighbor] = current
			heapq.heappush(open_heap, (new_distance, neighbor))

	return []


def path_total_cost(path: list[Position], terrain: np.ndarray) -> float:
	if len(path) <= 1:
		return 0.0

	cost = 0.0
	for index in range(1, len(path)):
		cost += movement_step_cost(path[index - 1], path[index], terrain)
	return cost


def find_path(
	start: Position,
	goal: Position,
	terrain: np.ndarray,
	algorithm: str = "a_star",
	allow_diagonal: bool = False,
) -> list[Position]:
	if algorithm == "a_star":
		return a_star_path(start, goal, terrain, allow_diagonal=allow_diagonal)
	if algorithm == "dijkstra":
		return dijkstra_path(start, goal, terrain, allow_diagonal=allow_diagonal)

	raise ValueError(f"Unsupported pathfinding algorithm: {algorithm}")


def clamp(value: float, minimum: float, maximum: float) -> float:
	return max(minimum, min(value, maximum))


def cover_ratio(
	position: Position,
	terrain: np.ndarray,
	include_diagonal: bool = True,
) -> float:
	x, y = position
	neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
	if include_diagonal:
		neighbors.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

	blocked = 0
	for dx, dy in neighbors:
		nx = x + dx
		ny = y + dy
		if is_position_in_bounds((nx, ny), terrain) and terrain[ny, nx] == 1:
			blocked += 1

	return blocked / len(neighbors)


def directional_cover_ratio(
	position: Position,
	attacker_position: Position,
	terrain: np.ndarray,
	include_diagonal: bool = True,
) -> float:
	regular = cover_ratio(position, terrain, include_diagonal)
	if regular == 0.0:
		return 0.0

	x, y = position
	ax, ay = attacker_position
	dx_att = ax - x
	dy_att = ay - y

	neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
	if include_diagonal:
		neighbors.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

	obstacles_between = 0
	total_obstacles = 0
	for dx, dy in neighbors:
		nx = x + dx
		ny = y + dy
		if not is_position_in_bounds((nx, ny), terrain):
			continue
		if terrain[ny, nx] == 1:
			total_obstacles += 1
			dot = dx * dx_att + dy * dy_att
			if dot > 0:
				obstacles_between += 1

	if total_obstacles == 0:
		return 0.0

	directional_factor = obstacles_between / total_obstacles
	return regular * directional_factor


def hit_probability(
	distance: float,
	attack_range: float,
	base_accuracy: float,
	cover: float,
	attacker_cover: float = 0.0,
) -> float:
	if attack_range <= 0:
		return 0.0
	if distance > attack_range:
		return 0.0

	normalized_distance = clamp(distance / attack_range, 0.0, 1.0)
	distance_penalty = 0.45 * normalized_distance
	cover_penalty = 0.50 * clamp(cover, 0.0, 1.0)
	stability_bonus = 0.12 * clamp(attacker_cover, 0.0, 1.0)

	chance = base_accuracy - distance_penalty - cover_penalty + stability_bonus
	return clamp(chance, 0.0, 1.0)


def calculate_damage(
	firepower: float,
	distance: float,
	attack_range: float,
	cover: float,
	armor: float = 0.0,
	flanking_multiplier: float = 1.0,
) -> int:
	if firepower <= 0:
		return 0
	if attack_range <= 0:
		return 0

	normalized_distance = clamp(distance / attack_range, 0.0, 1.0)
	range_factor = 1.0 - (0.50 * normalized_distance)
	cover_factor = 1.0 - (0.55 * clamp(cover, 0.0, 1.0))
	armor_factor = 1.0 - clamp(armor, 0.0, 0.90)

	raw_damage = (
		firepower * range_factor * cover_factor * armor_factor * flanking_multiplier
	)
	return max(1, int(round(raw_damage)))


def bresenham_line(x0: int, y0: int, x1: int, y1: int):
	dx = abs(x1 - x0)
	dy = abs(y1 - y0)
	x, y = x0, y0
	sx = -1 if x0 > x1 else 1
	sy = -1 if y0 > y1 else 1

	if dx > dy:
		err = dx / 2.0
		while x != x1:
			yield x, y
			err -= dy
			if err < 0:
				y += sy
				err += dx
			x += sx
	else:
		err = dy / 2.0
		while y != y1:
			yield x, y
			err -= dx
			if err < 0:
				x += sx
				err += dy
			y += sy
	yield x, y


def has_line_of_sight(
	pos1: tuple[int, int], pos2: tuple[int, int], terrain: np.ndarray
) -> bool:
	if not is_position_in_bounds(pos1, terrain) or not is_position_in_bounds(
		pos2, terrain
	):
		return False

	x0, y0 = pos1
	x1, y1 = pos2
	line = list(bresenham_line(x0, y0, x1, y1))
	for x, y in line[1:-1]:
		if terrain[y, x] == 1:
			return False
	return True


def is_in_vision_cone(
	observer_pos: tuple[int, int],
	target_pos: tuple[int, int],
	facing_direction: tuple[int, int],
	view_angle_deg: float,
) -> bool:
	if observer_pos == target_pos:
		return True

	fx, fy = facing_direction
	facing_norm = math.hypot(fx, fy)
	if facing_norm == 0:
		return True

	tx = target_pos[0] - observer_pos[0]
	ty = target_pos[1] - observer_pos[1]
	target_norm = math.hypot(tx, ty)
	if target_norm == 0:
		return True

	cos_theta = (fx * tx + fy * ty) / (facing_norm * target_norm)
	cos_theta = max(-1.0, min(1.0, cos_theta))
	limit = math.cos(math.radians(view_angle_deg / 2))
	return cos_theta >= limit


def detection_score(
	observer_pos: tuple[int, int],
	target_pos: tuple[int, int],
	terrain: np.ndarray,
	observation_range: float,
	facing_direction: tuple[int, int],
	view_angle_deg: float,
) -> float:
	distance = euclidean_distance(observer_pos, target_pos)
	if distance > observation_range:
		return 0.0

	if not is_in_vision_cone(
		observer_pos=observer_pos,
		target_pos=target_pos,
		facing_direction=facing_direction,
		view_angle_deg=view_angle_deg,
	):
		return 0.0

	if not has_line_of_sight(observer_pos, target_pos, terrain):
		return 0.0

	if observation_range <= 0:
		return 0.0
	return max(0.0, 1.0 - (distance / observation_range))

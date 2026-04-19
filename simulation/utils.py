import math

import numpy as np


def is_position_in_bounds(pos: tuple[int, int], terrain: np.ndarray) -> bool:
	x, y = pos
	return 0 <= y < terrain.shape[0] and 0 <= x < terrain.shape[1]


def euclidean_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> float:
	dx = pos1[0] - pos2[0]
	dy = pos1[1] - pos2[1]
	return math.hypot(dx, dy)


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

	# Start and end cells are excluded to allow checking visibility to occupied cells.
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


def visible_cells(
	observer_pos: tuple[int, int],
	terrain: np.ndarray,
	observation_range: float,
	facing_direction: tuple[int, int],
	view_angle_deg: float,
) -> set[tuple[int, int]]:
	visible: set[tuple[int, int]] = set()
	for y in range(terrain.shape[0]):
		for x in range(terrain.shape[1]):
			pos = (x, y)
			if (
				detection_score(
					observer_pos=observer_pos,
					target_pos=pos,
					terrain=terrain,
					observation_range=observation_range,
					facing_direction=facing_direction,
					view_angle_deg=view_angle_deg,
				)
				> 0
			):
				visible.add(pos)
	return visible

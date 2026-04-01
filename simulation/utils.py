import numpy as np


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
	x0, y0 = pos1
	x1, y1 = pos2

	for x, y in bresenham_line(x0, y0, x1, y1):
		# Skip exact start/end as they're occupied by agents.
		# For now, assume any tile that is 1 blocks sight.
		if terrain[y, x] == 1:
			return False
	return True

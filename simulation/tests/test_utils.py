import numpy as np

from simulation.utils import has_line_of_sight


def test_has_line_of_sight_clear() -> None:
	terrain = np.zeros((10, 10), dtype=int)
	assert has_line_of_sight((1, 1), (1, 8), terrain) is True
	assert has_line_of_sight((1, 1), (8, 8), terrain) is True


def test_has_line_of_sight_blocked() -> None:
	terrain = np.zeros((10, 10), dtype=int)
	terrain[4, 4] = 1
	terrain[4, 5] = 1
	terrain[4, 6] = 1
	assert has_line_of_sight((4, 2), (4, 8), terrain) is False
	assert has_line_of_sight((2, 5), (3, 5), terrain) is True

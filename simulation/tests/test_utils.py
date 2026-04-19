import numpy as np

from simulation.utils import detection_score, has_line_of_sight, is_in_vision_cone


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


def test_has_line_of_sight_ignores_endpoints() -> None:
	terrain = np.zeros((10, 10), dtype=int)
	terrain[1, 1] = 1
	terrain[8, 1] = 1
	assert has_line_of_sight((1, 1), (1, 8), terrain) is True


def test_is_in_vision_cone_directional() -> None:
	observer = (5, 5)
	assert is_in_vision_cone(observer, (5, 8), (0, 1), 90.0) is True
	assert is_in_vision_cone(observer, (5, 2), (0, 1), 90.0) is False


def test_detection_score_falls_with_distance() -> None:
	terrain = np.zeros((10, 10), dtype=int)
	near_score = detection_score((2, 2), (2, 3), terrain, 6, (0, 1), 120.0)
	far_score = detection_score((2, 2), (2, 7), terrain, 6, (0, 1), 120.0)

	assert near_score > far_score
	assert near_score > 0
	assert far_score > 0


def test_detection_score_zero_without_los() -> None:
	terrain = np.zeros((10, 10), dtype=int)
	terrain[3, 2] = 1
	score = detection_score((2, 2), (2, 5), terrain, 6, (0, 1), 120.0)
	assert score == 0.0

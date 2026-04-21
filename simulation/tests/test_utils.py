import numpy as np

from simulation.utils import (
	a_star_path,
	calculate_damage,
	cover_ratio,
	detection_score,
	dijkstra_path,
	find_path,
	has_line_of_sight,
	hit_probability,
	is_in_vision_cone,
	path_total_cost,
)


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


def test_a_star_path_avoids_obstacles() -> None:
	terrain = np.zeros((7, 7), dtype=int)
	terrain[2, 2] = 1
	terrain[3, 2] = 1
	terrain[4, 2] = 1

	path = a_star_path((1, 3), (5, 3), terrain, allow_diagonal=False)

	assert path[0] == (1, 3)
	assert path[-1] == (5, 3)
	assert (2, 3) not in path


def test_dijkstra_path_prefers_lower_topography_cost() -> None:
	terrain = np.zeros((5, 5), dtype=int)
	terrain[2, 1] = 6
	terrain[2, 2] = 6
	terrain[2, 3] = 6

	path = dijkstra_path((0, 2), (4, 2), terrain, allow_diagonal=False)

	assert path[0] == (0, 2)
	assert path[-1] == (4, 2)
	assert (1, 2) not in path
	assert (2, 2) not in path
	assert (3, 2) not in path

	direct_path = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
	assert path_total_cost(path, terrain) < path_total_cost(direct_path, terrain)


def test_find_path_returns_empty_for_unreachable_goal() -> None:
	terrain = np.zeros((5, 5), dtype=int)
	terrain[2, 1] = 1
	terrain[2, 2] = 1
	terrain[2, 3] = 1
	terrain[1, 2] = 1
	terrain[3, 2] = 1

	path = find_path((0, 2), (2, 2), terrain, algorithm="a_star")
	assert path == []


def test_cover_ratio_increases_near_obstacles() -> None:
	terrain = np.zeros((7, 7), dtype=int)
	terrain[2, 2] = 1
	terrain[2, 3] = 1
	terrain[2, 4] = 1

	low_cover = cover_ratio((1, 1), terrain)
	high_cover = cover_ratio((3, 3), terrain)

	assert high_cover > low_cover


def test_hit_probability_decreases_with_distance_and_cover() -> None:
	close_open = hit_probability(1.0, 5.0, 0.8, 0.0)
	far_open = hit_probability(5.0, 5.0, 0.8, 0.0)
	close_covered = hit_probability(1.0, 5.0, 0.8, 0.8)

	assert close_open > far_open
	assert close_open > close_covered


def test_calculate_damage_reduced_by_cover_and_armor() -> None:
	base = calculate_damage(20, 1.0, 5.0, 0.0, 0.0)
	covered = calculate_damage(20, 1.0, 5.0, 1.0, 0.0)
	armored = calculate_damage(20, 1.0, 5.0, 0.0, 0.5)

	assert base > covered
	assert base > armored

from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure

from simulation.gui.portrayal import agent_portrayal, draw_terrain
from simulation.gui.theme import (
	STATE_COLORS,
	class_marker,
	state_edge_color,
	team_color,
)


def _model(terrain, obstacle_hp=None, obstacle_max_hp=1.0) -> SimpleNamespace:
	return SimpleNamespace(
		terrain=terrain,
		obstacle_hp=obstacle_hp,
		obstacle_max_hp=obstacle_max_hp,
	)


def test_team_color_known_and_unknown() -> None:
	assert team_color("Blue") == "#1f6feb"
	assert team_color("Red") == "#d1242f"
	assert team_color(None) == "#8250df"


def test_class_marker_known_and_default() -> None:
	assert class_marker("MainBattleTank") == "D"
	assert class_marker("ReconSquad") == "^"
	assert class_marker("UnknownClass") == "o"


def test_state_edge_color_respects_toggle() -> None:
	assert state_edge_color("retreat", show_state=True) == STATE_COLORS["retreat"]
	assert state_edge_color("retreat", show_state=False) == "black"
	assert state_edge_color(None, show_state=True) == "black"


def test_agent_portrayal_uses_team_class_and_state() -> None:
	agent = SimpleNamespace(
		team="Blue",
		ai_state="retreat",
		hp=50,
		max_hp=100,
		agent_class="ReconSquad",
	)

	style = agent_portrayal(agent, show_state=True)

	assert style.color == "#1f6feb"
	assert style.marker == "^"
	assert style.edgecolors == STATE_COLORS["retreat"]
	assert 60 < style.size < 200


def test_draw_terrain_adds_patch_per_terrain_cell() -> None:
	terrain = np.zeros((5, 5), dtype=int)
	terrain[1, 1] = 1
	terrain[2, 3] = 2

	fig = Figure()
	ax = fig.add_subplot()
	draw_terrain(ax, _model(terrain))

	assert len(ax.patches) == 2


def test_draw_terrain_draws_grid_lines() -> None:
	terrain = np.zeros((4, 4), dtype=int)

	fig = Figure()
	ax = fig.add_subplot()
	draw_terrain(ax, _model(terrain), include_grid_lines=True)

	assert len(ax.lines) == (4 + 1) * 2


def test_draw_terrain_without_terrain_is_noop() -> None:
	fig = Figure()
	ax = fig.add_subplot()
	draw_terrain(ax, _model(None))

	assert len(ax.patches) == 0

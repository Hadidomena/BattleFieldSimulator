from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

from simulation.gui.theme import (
	OBSTACLE_DAMAGED_COLOR,
	OBSTACLE_INTACT_COLOR,
	ROUGH_TERRAIN_COLOR,
	class_marker,
	state_edge_color,
	team_color,
)

if TYPE_CHECKING:
	from mesa.visualization.components import AgentPortrayalStyle


def _mix(color_a: str, color_b: str, ratio: float) -> tuple[float, float, float]:
	ratio = max(0.0, min(1.0, ratio))
	rgba = mcolors.to_rgb(color_a)
	rgb_b = mcolors.to_rgb(color_b)
	return tuple(a + (b - a) * ratio for a, b in zip(rgba, rgb_b, strict=True))


def agent_portrayal(agent, show_state: bool = False) -> AgentPortrayalStyle:
	from mesa.visualization.components import AgentPortrayalStyle  # noqa: PLC0415

	color = team_color(getattr(agent, "team", None))
	class_name = getattr(agent, "agent_class", None) or type(agent).__name__
	marker = class_marker(class_name)

	hp = getattr(agent, "hp", 0)
	max_hp = max(1, getattr(agent, "max_hp", 1))
	hp_ratio = max(0.0, min(1.0, hp / max_hp))

	edge_color = state_edge_color(getattr(agent, "ai_state", None), show_state)

	return AgentPortrayalStyle(
		color=color,
		marker=marker,
		size=60 + 140 * hp_ratio,
		alpha=0.45 + 0.55 * hp_ratio,
		edgecolors=edge_color,
		linewidths=1.8 if show_state else 0.8,
		zorder=2,
	)


def draw_terrain(ax, model, include_grid_lines: bool = False) -> None:
	terrain = getattr(model, "terrain", None)
	if terrain is None:
		return

	obstacle_hp = getattr(model, "obstacle_hp", None)
	obstacle_max_hp = getattr(model, "obstacle_max_hp", 0.0) or 1.0

	height, width = terrain.shape
	for row in range(height):
		for col in range(width):
			value = int(terrain[row, col])
			if value == 1:
				ratio = 1.0
				if obstacle_hp is not None:
					ratio = obstacle_hp[row, col] / obstacle_max_hp
				color = _mix(OBSTACLE_DAMAGED_COLOR, OBSTACLE_INTACT_COLOR, ratio)
			elif value > 1:
				color = ROUGH_TERRAIN_COLOR
			else:
				continue

			ax.add_patch(
				Rectangle(
					(col - 0.5, row - 0.5),
					1,
					1,
					facecolor=color,
					edgecolor="none",
					zorder=0.5,
				)
			)

	ax.set_xlim(-0.5, width - 0.5)
	ax.set_ylim(height - 0.5, -0.5)
	ax.set_aspect("equal", adjustable="box")

	if include_grid_lines:
		for col in range(width + 1):
			ax.axvline(col - 0.5, color="gray", linestyle=":", linewidth=0.5, zorder=1)
		for row in range(height + 1):
			ax.axhline(row - 0.5, color="gray", linestyle=":", linewidth=0.5, zorder=1)

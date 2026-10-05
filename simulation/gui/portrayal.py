from __future__ import annotations

from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from mesa.visualization.components import AgentPortrayalStyle

TEAM_COLORS = {
	"Blue": "#1f6feb",
	"Red": "#d1242f",
}
UNKNOWN_TEAM_COLOR = "#8250df"

CLASS_MARKERS = {
	"InfantrySquad": "o",
	"ReconSquad": "^",
	"MechanizedInfantry": "s",
	"MainBattleTank": "D",
}
DEFAULT_MARKER = "o"

OBSTACLE_INTACT_COLOR = "#3d4451"
OBSTACLE_DAMAGED_COLOR = "#9aa3b2"
ROUGH_TERRAIN_COLOR = "#e6d8b5"


def _mix(color_a: str, color_b: str, ratio: float) -> tuple[float, float, float]:
	ratio = max(0.0, min(1.0, ratio))
	rgba = mcolors.to_rgb(color_a)
	rgb_b = mcolors.to_rgb(color_b)
	return tuple(a + (b - a) * ratio for a, b in zip(rgba, rgb_b, strict=True))


def agent_portrayal(agent) -> AgentPortrayalStyle:
	team = getattr(agent, "team", None)
	color = TEAM_COLORS.get(team, UNKNOWN_TEAM_COLOR)
	marker = CLASS_MARKERS.get(type(agent).__name__, DEFAULT_MARKER)

	hp = getattr(agent, "hp", 0)
	max_hp = max(1, getattr(agent, "max_hp", 1))
	hp_ratio = max(0.0, min(1.0, hp / max_hp))

	return AgentPortrayalStyle(
		color=color,
		marker=marker,
		size=60 + 140 * hp_ratio,
		alpha=0.45 + 0.55 * hp_ratio,
		edgecolors="black",
		linewidths=0.8,
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

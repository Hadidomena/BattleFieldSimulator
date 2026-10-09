from __future__ import annotations

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

STATE_COLORS = {
	"advance": "#2e7d32",
	"engage": "#e65100",
	"retreat": "#6a1b9a",
	"patrol": "#00838f",
}
STATE_ORDER = ["advance", "engage", "retreat", "patrol"]

OBSTACLE_INTACT_COLOR = "#3d4451"
OBSTACLE_DAMAGED_COLOR = "#9aa3b2"
ROUGH_TERRAIN_COLOR = "#e6d8b5"


def team_color(team: str | None) -> str:
	return TEAM_COLORS.get(team, UNKNOWN_TEAM_COLOR)


def class_marker(class_name: str) -> str:
	return CLASS_MARKERS.get(class_name, DEFAULT_MARKER)


def state_edge_color(state: str | None, show_state: bool) -> str:
	if not show_state:
		return "black"
	return STATE_COLORS.get(state, "black")

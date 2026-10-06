from simulation.gui.theme import (
	STATE_COLORS,
	class_marker,
	state_edge_color,
	team_color,
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

import numpy as np
import pandas as pd
import reacton

from simulation.gui import app
from simulation.gui.model_factory import SimulationModel
from simulation.gui.replay import ReplayData


def _model() -> SimulationModel:
	return SimulationModel(blue_count=1, red_count=1, seed="1")


def _replay() -> ReplayData:
	agent_records = pd.DataFrame(
		[
			{
				"step": 0,
				"unique_id": 1,
				"agent_class": "InfantrySquad",
				"team": "Blue",
				"pos_x": 1,
				"pos_y": 1,
				"hp": 100,
				"max_hp": 100,
				"ai_state": "advance",
			}
		]
	)
	model_records = pd.DataFrame(
		[
			{
				"step": 0,
				"alive_blue": 1,
				"alive_red": 1,
				"kills_blue": 0,
				"kills_red": 0,
				"damage_blue": 0,
				"damage_red": 0,
			}
		]
	)
	return ReplayData(
		scenario="s",
		run="run_000",
		terrain=np.zeros((4, 4), dtype=int),
		agent_records=agent_records,
		model_records=model_records,
		event_records=pd.DataFrame(),
		steps=[0],
	)


def test_gui_app_exposes_page_component() -> None:
	assert app.Page is not None
	assert app._ReplayMap is not None
	assert app._LegendView is not None


def test_legend_view_renders() -> None:
	widget, rc = reacton.render(app._LegendView())
	assert widget is not None
	rc.close()


def test_space_view_renders() -> None:
	widget, rc = reacton.render(app._SpaceView(_model(), show_state=True))
	assert widget is not None
	rc.close()


def test_state_chart_renders() -> None:
	widget, rc = reacton.render(app._StateChart(_model()))
	assert widget is not None
	rc.close()


def test_replay_map_renders() -> None:
	widget, rc = reacton.render(app._ReplayMap(_replay(), 0, show_state=True))
	assert widget is not None
	rc.close()


def test_replay_chart_renders() -> None:
	widget, rc = reacton.render(app._ReplayChart(_replay(), 0))
	assert widget is not None
	rc.close()


def test_replay_details_renders() -> None:
	widget, rc = reacton.render(app._ReplayDetails(_replay(), 0))
	assert widget is not None
	rc.close()

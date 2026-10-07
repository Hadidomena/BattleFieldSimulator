def test_gui_app_exposes_page_component() -> None:
	from simulation.gui import app

	assert app.Page is not None
	assert app._ReplayMap is not None
	assert app._LegendView is not None

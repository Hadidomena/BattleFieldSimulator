from __future__ import annotations

from collections.abc import Callable

import solara
from matplotlib.figure import Figure
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.solara_viz import (
	ModelController,
	ModelCreator,
	ShowSteps,
)
from mesa.visualization.utils import update_counter

from simulation.gui.model_factory import SimulationModel
from simulation.gui.portrayal import agent_portrayal, draw_terrain

AGENT_STYLE_FIELDS = {
	"size": 50,
	"marker": "o",
	"zorder": 1,
	"alpha": 1.0,
	"linewidths": 1.0,
}


def _figure_to_png(fig: Figure) -> bytes:
	import io

	buf = io.BytesIO()
	fig.savefig(buf, format="png", bbox_inches="tight")
	buf.seek(0)
	return buf.getvalue()


@solara.component
def _SpaceView(model):
	update_counter.get()

	fig = Figure(constrained_layout=True, figsize=(6, 6))
	ax = fig.add_subplot()

	draw_terrain(ax, model, include_grid_lines=True)
	_draw_agents(ax, model)

	solara.Image(_figure_to_png(fig), width="100%")


def _draw_agents(ax, model) -> None:
	space = getattr(model, "grid", None) or getattr(model, "space", None)
	agents = list(space.agents)

	by_zorder: dict[int, list] = {}
	for agent in agents:
		style = agent_portrayal(agent)
		if style.x is None or style.y is None:
			style.x, style.y = agent.pos
		by_zorder.setdefault(style.zorder or 1, []).append(style)

	for zorder in sorted(by_zorder):
		styles: list[AgentPortrayalStyle] = by_zorder[zorder]
		xs = [s.x for s in styles]
		ys = [s.y for s in styles]
		colors = [s.color for s in styles]
		sizes = [s.size or AGENT_STYLE_FIELDS["size"] for s in styles]
		alphas = [s.alpha for s in styles]
		markers = [s.marker or AGENT_STYLE_FIELDS["marker"] for s in styles]

		for marker in set(markers):
			mask = [m == marker for m in markers]
			ax.scatter(
				[x for x, keep in zip(xs, mask, strict=True) if keep],
				[y for y, keep in zip(ys, mask, strict=True) if keep],
				s=[s for s, keep in zip(sizes, mask, strict=True) if keep],
				c=[c for c, keep in zip(colors, mask, strict=True) if keep],
				alpha=[a for a, keep in zip(alphas, mask, strict=True) if keep],
				marker=marker,
				edgecolors="black",
				linewidths=0.8,
				zorder=zorder,
			)


def _plot_component(measures: str | list[str]) -> Callable:
	measures_list = [measures] if isinstance(measures, str) else measures

	@solara.component
	def Plot(model):
		update_counter.get()
		fig = Figure(constrained_layout=True, figsize=(6, 3))
		ax = fig.subplots()
		df = model.datacollector.get_model_vars_dataframe()
		for measure in measures_list:
			if measure in df.columns:
				ax.plot(df[measure], label=measure)
		ax.set_xlabel("Step")
		if measures_list:
			ax.legend(loc="best")
		return solara.Image(_figure_to_png(fig), width="100%")

	return Plot


@solara.component
def Page():
	model = solara.use_memo(lambda: SimulationModel(), [])
	model = solara.use_reactive(model)

	with solara.AppBar():
		solara.AppBarTitle("Battlefield Simulator")
		solara.lab.ThemeToggle()

	with solara.Columns([0, 1], style={"width": "100%", "padding": "12px"}):
		with solara.Column(style={"min-width": "340px", "max-width": "400px"}):
			with solara.Card("Controls"):
				ModelController(
					model,
					play_interval=solara.use_reactive(250),
					render_interval=solara.use_reactive(1),
					use_threads=solara.use_reactive(False),
				)
			with solara.Card("Model Parameters"):
				ModelCreator(model, {})
			with solara.Card("Information"):
				ShowSteps(model.value)

		with solara.Column(style={"min-width": "0"}):
			with solara.Columns([3, 2]):
				with solara.Card("Battlefield"):
					_SpaceView(model.value)
				with solara.Column():
					with solara.Card("Population"):
						_plot_component(["Alive_Blue", "Alive_Red"])(model.value)
					with solara.Card("Damage"):
						_plot_component(["Damage_Blue", "Damage_Red"])(model.value)
					with solara.Card("Destroyed Obstacles"):
						_plot_component(["Destroyed_Obstacles"])(model.value)

from __future__ import annotations

from collections.abc import Callable

import solara
from matplotlib.figure import Figure
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.solara_viz import ModelController, ShowSteps
from mesa.visualization.utils import update_counter

from simulation.gui.model_factory import (
	CUSTOM_SCENARIO,
	DEFAULT_MAP,
	DEFAULT_UNIT_CLASS,
	SimulationModel,
	default_model_parameters,
	unit_class_names,
)
from simulation.gui.portrayal import agent_portrayal, draw_terrain
from simulation.gui.replay import (
	ReplayData,
	TerrainModel,
	list_replay_scenarios,
	list_runs,
	load_run,
	record_run,
)
from simulation.gui.theme import (
	CLASS_MARKERS,
	DEFAULT_MARKER,
	STATE_COLORS,
	STATE_ORDER,
	TEAM_COLORS,
)
from simulation.scenarios import list_map_names, list_scenario_names

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
def _SpaceView(model, show_state: bool):
	update_counter.get()

	fig = Figure(constrained_layout=True, figsize=(6, 6))
	ax = fig.add_subplot()

	draw_terrain(ax, model, include_grid_lines=True)
	_draw_agents(ax, _live_agents(model), show_state=show_state)

	solara.Image(_figure_to_png(fig), width="100%")


def _live_agents(model) -> list:
	space = getattr(model, "grid", None) or getattr(model, "space", None)
	return list(space.agents) if space is not None else []


def _draw_agents(ax, agents, show_state: bool = False) -> None:
	by_zorder: dict[int, list] = {}
	for agent in agents:
		style = agent_portrayal(agent, show_state=show_state)
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
		edges = [s.edgecolors or "black" for s in styles]
		widths = [s.linewidths if s.linewidths is not None else 0.8 for s in styles]

		for marker in set(markers):
			mask = [m == marker for m in markers]
			ax.scatter(
				[x for x, keep in zip(xs, mask, strict=True) if keep],
				[y for y, keep in zip(ys, mask, strict=True) if keep],
				s=[s for s, keep in zip(sizes, mask, strict=True) if keep],
				c=[c for c, keep in zip(colors, mask, strict=True) if keep],
				alpha=[a for a, keep in zip(alphas, mask, strict=True) if keep],
				marker=marker,
				edgecolors=[e for e, keep in zip(edges, mask, strict=True) if keep],
				linewidths=[w for w, keep in zip(widths, mask, strict=True) if keep],
				zorder=zorder,
			)


@solara.component
def _StateChart(model):
	update_counter.get()

	counts = {team: dict.fromkeys(STATE_ORDER, 0) for team in ("Blue", "Red")}
	for agent in model.agents:
		if getattr(agent, "hp", 0) <= 0:
			continue
		team = getattr(agent, "team", None)
		state = getattr(agent, "ai_state", None)
		if team in counts and state in counts[team]:
			counts[team][state] += 1

	fig = Figure(constrained_layout=True, figsize=(6, 3))
	ax = fig.subplots()
	positions = list(range(len(STATE_ORDER)))
	bar_width = 0.38
	for offset, team in ((-bar_width / 2, "Blue"), (bar_width / 2, "Red")):
		values = [counts[team][state] for state in STATE_ORDER]
		ax.bar(
			[p + offset for p in positions],
			values,
			width=bar_width,
			label=team,
			color=TEAM_COLORS[team],
		)
	ax.set_xticks(positions)
	ax.set_xticklabels(STATE_ORDER)
	ax.set_ylabel("Units")
	ax.legend(loc="best")
	return solara.Image(_figure_to_png(fig), width="100%")


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
def _ConfigEditor(model, model_parameters, error_message):
	values = model_parameters.value

	def update(name, value):
		new_parameters = {**model_parameters.value, name: value}
		model_parameters.set(new_parameters)
		try:
			model.set(type(model.value)(**new_parameters))
			error_message.set(None)
		except (ValueError, FileNotFoundError, KeyError) as error:
			error_message.set(str(error))

	solara.Select(
		label="Scenario",
		value=values.get("scenario_name", CUSTOM_SCENARIO),
		values=[CUSTOM_SCENARIO, *list_scenario_names()],
		on_value=lambda value: update("scenario_name", value),
	)
	solara.Select(
		label="Map (Custom scenario)",
		value=values.get("map_name", DEFAULT_MAP),
		values=list_map_names(),
		on_value=lambda value: update("map_name", value),
	)
	solara.Select(
		label="Blue unit class",
		value=values.get("blue_units", DEFAULT_UNIT_CLASS),
		values=unit_class_names(),
		on_value=lambda value: update("blue_units", value),
	)
	solara.Select(
		label="Red unit class",
		value=values.get("red_units", DEFAULT_UNIT_CLASS),
		values=unit_class_names(),
		on_value=lambda value: update("red_units", value),
	)
	solara.SliderInt(
		label="Blue unit count",
		value=values.get("blue_count", 3),
		min=1,
		max=10,
		on_value=lambda value: update("blue_count", value),
	)
	solara.SliderInt(
		label="Red unit count",
		value=values.get("red_count", 3),
		min=1,
		max=10,
		on_value=lambda value: update("red_count", value),
	)
	solara.InputText(
		label="Seed (blank = random)",
		value=str(values.get("seed", "")),
		on_value=lambda value: update("seed", value),
	)
	solara.SliderInt(
		label="Max steps (0 = no limit)",
		value=values.get("max_steps", 0),
		min=0,
		max=500,
		step=10,
		on_value=lambda value: update("max_steps", value),
	)


@solara.component
def _LegendView():
	classes = unit_class_names()

	fig = Figure(constrained_layout=True, figsize=(4.5, 3.4))
	ax = fig.add_subplot()
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_axis_off()

	ax.text(0.02, 0.98, "Unit", fontsize=9, fontweight="bold", va="top")
	for index, name in enumerate(classes):
		y = 0.9 - index * 0.1
		marker = CLASS_MARKERS.get(name, DEFAULT_MARKER)
		ax.scatter(
			[0.06],
			[y],
			marker=marker,
			s=110,
			c="#555555",
			edgecolors="black",
			linewidths=0.8,
		)
		ax.text(0.15, y, name, va="center", ha="left", fontsize=7.5)

	ax.text(0.52, 0.98, "Team", fontsize=9, fontweight="bold", va="top")
	for index, (team, color) in enumerate(TEAM_COLORS.items()):
		y = 0.9 - index * 0.1
		ax.scatter(
			[0.56],
			[y],
			marker="o",
			s=110,
			c=color,
			edgecolors="black",
			linewidths=0.8,
		)
		ax.text(0.65, y, team, va="center", ha="left", fontsize=7.5)

	ax.text(0.52, 0.62, "AI state", fontsize=9, fontweight="bold", va="top")
	for index, state in enumerate(STATE_ORDER):
		y = 0.54 - index * 0.1
		ax.scatter(
			[0.56],
			[y],
			marker="o",
			s=110,
			c="white",
			edgecolors=STATE_COLORS[state],
			linewidths=2.0,
		)
		ax.text(0.65, y, state, va="center", ha="left", fontsize=7.5)

	solara.Image(_figure_to_png(fig), width="100%")


@solara.component
def _ReplayMap(replay: ReplayData, step: int, show_state: bool):
	fig = Figure(constrained_layout=True, figsize=(6, 6))
	ax = fig.add_subplot()

	draw_terrain(ax, TerrainModel(replay.terrain), include_grid_lines=True)
	_draw_agents(ax, replay.agents_at(step), show_state=show_state)

	solara.Image(_figure_to_png(fig), width="100%")


@solara.component
def _ReplayChart(replay: ReplayData, step: int):
	fig = Figure(constrained_layout=True, figsize=(6, 3))
	ax = fig.subplots()

	df = replay.model_records
	if not df.empty:
		if "alive_blue" in df.columns:
			ax.plot(
				df["step"], df["alive_blue"], label="Blue", color=TEAM_COLORS["Blue"]
			)
		if "alive_red" in df.columns:
			ax.plot(df["step"], df["alive_red"], label="Red", color=TEAM_COLORS["Red"])
		ax.axvline(step, color="gray", linestyle="--", linewidth=1)
		ax.legend(loc="best")

	ax.set_xlabel("Step")
	ax.set_ylabel("Alive")
	return solara.Image(_figure_to_png(fig), width="100%")


@solara.component
def _ReplayDetails(replay: ReplayData, step: int):
	record = replay.model_at(step)
	if record is None:
		solara.Text("No data for this step.")
		return

	solara.Markdown(
		f"**Step {step}**\n\n"
		f"- Alive: Blue {int(record.get('alive_blue', 0))} / "
		f"Red {int(record.get('alive_red', 0))}\n"
		f"- Kills: Blue {int(record.get('kills_blue', 0))} / "
		f"Red {int(record.get('kills_red', 0))}\n"
		f"- Damage: Blue {int(record.get('damage_blue', 0))} / "
		f"Red {int(record.get('damage_red', 0))}"
	)


@solara.component
def _LiveSidebar(model, model_parameters, error_message, show_states, on_record):
	with solara.Card("Controls"):
		ModelController(
			model,
			model_parameters=model_parameters,
			play_interval=solara.use_reactive(250),
			render_interval=solara.use_reactive(1),
			use_threads=solara.use_reactive(False),
		)
		solara.Checkbox(
			label="Show AI state outline",
			value=show_states,
			on_value=show_states.set,
		)
		solara.Button("Record run", on_click=on_record)
	with solara.Card("Model Parameters"):
		_ConfigEditor(model, model_parameters, error_message)
		if error_message.value:
			solara.Error(error_message.value)
	with solara.Card("Information"):
		ShowSteps(model.value)


@solara.component
def _ReplaySidebar(
	scenarios,
	selected_scenario,
	runs,
	selected_run,
	replay,
	step,
	on_scenario,
	on_run,
	on_step,
):
	with solara.Card("Replay"):
		if not scenarios:
			solara.Info(
				"No telemetry found. Run a scenario, or use 'Record run' "
				"in the Live view."
			)
			return
		solara.Select(
			label="Scenario",
			value=selected_scenario,
			values=scenarios,
			on_value=on_scenario,
		)
		if runs:
			solara.Select(
				label="Run",
				value=selected_run,
				values=runs,
				on_value=on_run,
			)
		if replay is not None:
			solara.SliderInt(
				label="Step",
				value=step,
				min=0,
				max=replay.max_step,
				on_value=on_step,
			)
			solara.Text(f"{len(replay.agents_at(step))} units on the field")


@solara.component
def _LiveMain(model, show_states):
	with solara.Columns([3, 2]):
		with solara.Card("Battlefield"):
			_SpaceView(model, show_states)
		with solara.Column():
			with solara.Card("Population"):
				_plot_component(["Alive_Blue", "Alive_Red"])(model)
			with solara.Card("Damage"):
				_plot_component(["Damage_Blue", "Damage_Red"])(model)
			with solara.Card("Destroyed Obstacles"):
				_plot_component(["Destroyed_Obstacles"])(model)
			with solara.Card("AI States"):
				_StateChart(model)


@solara.component
def _ReplayMain(replay, step, show_states):
	with solara.Columns([3, 2]):
		with solara.Card("Replay battlefield"):
			_ReplayMap(replay, step, show_states)
		with solara.Column():
			with solara.Card("Population"):
				_ReplayChart(replay, step)
			with solara.Card("Step details"):
				_ReplayDetails(replay, step)


@solara.component
def Page():
	model = solara.use_reactive(solara.use_memo(lambda: SimulationModel(), []))
	model_parameters = solara.use_reactive(solara.use_memo(default_model_parameters, []))
	error_message = solara.use_reactive(None)
	show_states = solara.use_reactive(True)
	view = solara.use_reactive("Live")
	replay_scenario = solara.use_reactive(None)
	replay_run = solara.use_reactive(None)
	replay_step = solara.use_reactive(0)
	record_counter = solara.use_reactive(0)

	_ = record_counter.value
	scenarios = list_replay_scenarios()
	selected_scenario = (
		replay_scenario.value
		if replay_scenario.value in scenarios
		else (scenarios[0] if scenarios else None)
	)
	runs = list_runs(selected_scenario) if selected_scenario else []
	selected_run = (
		replay_run.value if replay_run.value in runs else (runs[0] if runs else None)
	)
	replay = solara.use_memo(
		lambda: (
			load_run(selected_scenario, selected_run)
			if selected_scenario and selected_run
			else None
		),
		[selected_scenario, selected_run, record_counter.value],
	)
	step = min(replay_step.value, replay.max_step) if replay is not None else 0

	def select_scenario(value):
		replay_scenario.set(value)
		replay_run.set(None)
		replay_step.set(0)

	def select_run(value):
		replay_run.set(value)
		replay_step.set(0)

	def record():
		record_run(model.value)
		record_counter.set(record_counter.value + 1)

	with solara.AppBar():
		solara.AppBarTitle("Battlefield Simulator")
		solara.ToggleButtonsSingle(
			value=view, values=["Live", "Replay"], on_value=view.set
		)
		solara.lab.ThemeToggle()

	with solara.Columns([0, 1], style={"width": "100%", "padding": "12px"}):
		with solara.Column(style={"min-width": "340px", "max-width": "400px"}):
			if view.value == "Live":
				_LiveSidebar(model, model_parameters, error_message, show_states, record)
			else:
				_ReplaySidebar(
					scenarios,
					selected_scenario,
					runs,
					selected_run,
					replay,
					step,
					select_scenario,
					select_run,
					replay_step.set,
				)
			with solara.Card("Legend"):
				_LegendView()

		with solara.Column(style={"min-width": "0"}):
			if view.value == "Live":
				_LiveMain(model.value, show_states.value)
			elif replay is None:
				solara.Info("Select a run to replay.")
			else:
				_ReplayMain(replay, step, show_states.value)

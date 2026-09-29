"""
Usage:
    python run_scenarios.py                          # run all scenarios
    python run_scenarios.py --scenario scenario_01   # run specific scenario
    python run_scenarios.py --output results/        # custom output directory
    python run_scenarios.py --list                   # list available scenarios
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from simulation.agent import (
	CombatAgent,
	InfantrySquad,
	MainBattleTank,
	MechanizedInfantry,
	ReconSquad,
)
from simulation.model import BattlefieldModel

PROJECT_ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = PROJECT_ROOT / "data" / "scenarios"

UNIT_CLASSES: dict[str, type[CombatAgent]] = {
	"CombatAgent": CombatAgent,
	"InfantrySquad": InfantrySquad,
	"ReconSquad": ReconSquad,
	"MechanizedInfantry": MechanizedInfantry,
	"MainBattleTank": MainBattleTank,
}


def load_board(path: str) -> np.ndarray:
	full_path = PROJECT_ROOT / path
	if not full_path.exists():
		raise FileNotFoundError(f"Map file not found: {full_path}")
	try:
		board = np.loadtxt(full_path, delimiter=",", dtype=int)
	except Exception:
		board = np.loadtxt(full_path, dtype=int)
	return board


def load_scenarios(scenario_dir: Path, filters: list[str] | None = None) -> list[dict]:
	scenarios: list[dict] = []
	for filepath in sorted(scenario_dir.glob("scenario_*.json")):
		if filters:
			if not any(f in filepath.stem for f in filters):
				continue
		with open(filepath) as f:
			scenario = json.load(f)
		scenario["_filename"] = filepath.stem
		scenarios.append(scenario)
	return scenarios


def build_model_from_scenario(
	scenario: dict, seed: int | None = None
) -> BattlefieldModel:
	board = load_board(scenario["map"])
	blue_config = scenario["blue_team"]
	red_config = scenario["red_team"]

	blue_spawn = [tuple(sp) for sp in blue_config.get("spawn_points", [])]
	red_spawn = [tuple(sp) for sp in red_config.get("spawn_points", [])]

	model = BattlefieldModel(
		board,
		blue_spawn_points=blue_spawn,
		red_spawn_points=red_spawn,
		seed=seed,
	)

	_reconfigure_agents(model, "Blue", blue_config)
	_reconfigure_agents(model, "Red", red_config)

	model.telemetry.reset()
	model.telemetry.record_step(model)
	return model


def _reconfigure_agents(model: BattlefieldModel, team: str, team_config: dict) -> None:
	to_remove = [a for a in model.agents if getattr(a, "team", None) == team]

	for agent in to_remove:
		if agent.pos is not None:
			model.grid.remove_agent(agent)
		agent.remove()

	unit_defs = team_config.get("units", [])
	spawn_points = team_config.get("spawn_points", [])

	spawn_index = 0
	for unit_def in unit_defs:
		class_name = unit_def["class"]
		count = unit_def.get("count", 1)
		overrides = unit_def.get("overrides", {})

		agent_cls = UNIT_CLASSES[class_name]

		for _ in range(count):
			spawn_pos = (
				tuple(spawn_points[spawn_index])
				if spawn_index < len(spawn_points)
				else None
			)
			spawn_index += 1

			if spawn_pos is None:
				continue

			agent = agent_cls(model, team=team, **overrides)
			model.grid.place_agent(agent, spawn_pos)


def _print_step_debug(model: BattlefieldModel) -> None:
	descriptions = " | ".join(
		f"#{agent.unique_id} {type(agent).__name__}({agent.team}) "
		f"pos={agent.pos} hp={getattr(agent, 'hp', 0)} "
		f"state={getattr(agent, 'ai_state', None)}"
		for agent in model.agents
	)
	print(f"[step {model.steps}] {descriptions}")


def run_single_simulation(
	model: BattlefieldModel, steps: int, verbose: bool = False
) -> dict:
	stdout_target = io.StringIO() if not verbose else None

	with redirect_stdout(stdout_target) if not verbose else nullcontext():
		for _ in range(steps):
			if not model.running:
				break
			model.step()
			if verbose:
				_print_step_debug(model)

	blue_alive = sum(
		1
		for a in model.agents
		if getattr(a, "team", None) == "Blue" and getattr(a, "hp", 0) > 0
	)
	red_alive = sum(
		1
		for a in model.agents
		if getattr(a, "team", None) == "Red" and getattr(a, "hp", 0) > 0
	)

	return {
		"blue_alive": blue_alive,
		"red_alive": red_alive,
		"blue_kills": model.kills_by_team["Blue"],
		"red_kills": model.kills_by_team["Red"],
		"blue_damage": model.damage_by_team["Blue"],
		"red_damage": model.damage_by_team["Red"],
		"blue_shots": model.shots_by_team["Blue"],
		"red_shots": model.shots_by_team["Red"],
		"blue_hits": model.hits_by_team["Blue"],
		"red_hits": model.hits_by_team["Red"],
		"blue_eliminated": model.eliminated_by_team["Blue"],
		"red_eliminated": model.eliminated_by_team["Red"],
		"steps_run": model.steps,
		"model_telemetry": model.telemetry.model_records,
		"agent_telemetry": model.telemetry.agent_records,
		"event_telemetry": model.telemetry.event_records,
	}


def aggregate_results(runs: list[dict]) -> dict:
	n = len(runs)
	if n == 0:
		return {}

	blue_wins = sum(1 for r in runs if r["red_alive"] == 0 and r["blue_alive"] > 0)
	red_wins = sum(1 for r in runs if r["blue_alive"] == 0 and r["red_alive"] > 0)
	draws = n - blue_wins - red_wins

	keys = [
		"blue_alive",
		"red_alive",
		"blue_kills",
		"red_kills",
		"blue_damage",
		"red_damage",
		"blue_shots",
		"red_shots",
		"blue_hits",
		"red_hits",
		"blue_eliminated",
		"red_eliminated",
		"steps_run",
	]

	aggregated: dict[str, float] = {}
	for key in keys:
		values = [r[key] for r in runs]
		aggregated[f"avg_{key}"] = sum(values) / n
		aggregated[f"min_{key}"] = min(values)
		aggregated[f"max_{key}"] = max(values)

	aggregated["blue_win_rate"] = blue_wins / n
	aggregated["red_win_rate"] = red_wins / n
	aggregated["draw_rate"] = draws / n
	aggregated["run_count"] = n

	return aggregated


def run_scenario(
	scenario: dict,
	output_dir: Path,
	capture_telemetry: bool = True,
	verbose: bool = False,
	seed: int | None = None,
) -> dict:
	scenario_name = scenario["name"]
	run_count = scenario["run_count"]
	steps = scenario["steps"]
	filename = scenario.get("_filename", "unknown")

	scenario_output_dir = output_dir / filename
	scenario_output_dir.mkdir(parents=True, exist_ok=True)

	run_results: list[dict] = []

	for run_idx in range(run_count):
		run_seed = None if seed is None else seed + run_idx
		model = build_model_from_scenario(scenario, seed=run_seed)
		run_result = run_single_simulation(model, steps, verbose=verbose)
		run_result["run_index"] = run_idx
		run_results.append(run_result)

		if capture_telemetry:
			run_telemetry_dir = scenario_output_dir / f"run_{run_idx:03d}"
			model.telemetry.export_csv(run_telemetry_dir)
			model.telemetry.export_json(run_telemetry_dir)

	aggr = aggregate_results(run_results)

	aggr["scenario_name"] = scenario_name
	aggr["_filename"] = filename
	aggr["hypothesis"] = scenario.get("hypothesis", "")
	aggr["total_steps"] = steps
	aggr["timestamp"] = datetime.now(timezone.utc).isoformat()

	summary_lines = [f"\n{'=' * 60}", f"SCENARIO: {scenario_name}"]
	summary_lines.append(f"Runs: {run_count} x {steps} steps")
	summary_lines.append(
		f"Blue win rate: {aggr['blue_win_rate']:.0%} "
		f"({aggr['blue_win_rate'] * run_count:.0f}/{run_count})"  # noqa
	)
	summary_lines.append(
		f"Red win rate:  {aggr['red_win_rate']:.0%} "
		f"({aggr['red_win_rate'] * run_count:.0f}/{run_count})"  # noqa
	)
	if aggr["draw_rate"] > 0:
		summary_lines.append(f"Draws:         {aggr['draw_rate']:.0%}")
	summary_lines.append(f"Avg Blue kills: {aggr['avg_blue_kills']:.1f}")
	summary_lines.append(f"Avg Red kills:  {aggr['avg_red_kills']:.1f}")
	summary_lines.append(f"Avg Blue dmg:   {aggr['avg_blue_damage']:.0f}")
	summary_lines.append(f"Avg Red dmg:    {aggr['avg_red_damage']:.0f}")
	summary_lines.append(
		f"Blue hit rate:  {aggr['avg_blue_hits'] / max(1, aggr['avg_blue_shots']):.1%}"
	)
	summary_lines.append(
		f"Red hit rate:   {aggr['avg_red_hits'] / max(1, aggr['avg_red_shots']):.1%}"
	)
	summary_lines.append(f"Hypothesis: {aggr['hypothesis']}")
	summary_lines.append(f"{'=' * 60}\n")
	print("\n".join(summary_lines))

	with open(scenario_output_dir / "summary.json", "w") as f:
		json.dump(aggr, f, indent="\t")

	raw_results_for_json = []
	for r in run_results:
		summary_fields = {k: v for k, v in r.items() if not k.endswith("_telemetry")}
		raw_results_for_json.append(summary_fields)
	with open(scenario_output_dir / "runs_raw.json", "w") as f:
		json.dump(raw_results_for_json, f, indent="\t")

	return aggr


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run BattleField Simulator experimental scenarios."
	)
	parser.add_argument(
		"--scenario",
		type=str,
		default=None,
		action="append",
		help="Filter: run only scenarios whose filename contains this string. "
		"Repeat for multiple filters (OR logic).",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="data/scenarios/results",
		help="Output directory for telemetry and aggregated results.",
	)
	parser.add_argument(
		"--list",
		action="store_true",
		help="List available scenarios and exit.",
	)
	parser.add_argument(
		"--no-telemetry",
		action="store_true",
		help="Skip per-run telemetry export to save disk space.",
	)
	parser.add_argument(
		"--verbose",
		"-v",
		action="store_true",
		help="Show per-turn agent debug output during simulation runs.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Seed for reproducible runs. Each run uses seed + run index; "
		"omit for non-deterministic runs.",
	)
	args = parser.parse_args()

	scenarios = load_scenarios(SCENARIO_DIR, args.scenario)

	if args.list:
		if not scenarios:
			print("No scenarios found.")
			sys.exit(1)
		print(f"Found {len(scenarios)} scenario(s):\n")
		for s in scenarios:
			print(
				f"  {s['_filename']:50s} {s['name']}\n"
				f"    Hypothesis: {s['hypothesis']}\n"
				f"    Runs: {s['run_count']} x {s['steps']} steps\n"
			)
		return

	if not scenarios:
		filters_text = ", ".join(args.scenario) if args.scenario else "(none)"
		print(f"No scenarios found matching '{filters_text}' in {SCENARIO_DIR}")
		sys.exit(1)

	output_dir = Path(args.output)
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Running {len(scenarios)} scenario(s)")
	print(f"Output directory: {output_dir.resolve()}\n")

	all_summaries: list[dict] = []
	for scenario in scenarios:
		summary = run_scenario(
			scenario,
			output_dir,
			capture_telemetry=not args.no_telemetry,
			verbose=args.verbose,
			seed=args.seed,
		)
		all_summaries.append(summary)

	with open(output_dir / "all_summaries.json", "w") as f:
		json.dump(all_summaries, f, indent="\t")

	print(f"\nAll results saved to {output_dir.resolve()}/")


if __name__ == "__main__":
	main()

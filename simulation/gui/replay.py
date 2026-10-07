from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.scenarios import PROJECT_ROOT, load_board, load_scenario

RESULTS_DIR = PROJECT_ROOT / "data" / "scenarios" / "results"


@dataclass
class TerrainModel:
	"""Minimal model-like object exposing terrain for ``draw_terrain``."""

	terrain: np.ndarray
	obstacle_hp: np.ndarray | None = None
	obstacle_max_hp: float = 1.0


@dataclass
class ReplayAgent:
	team: str | None
	agent_class: str
	hp: float
	max_hp: float
	ai_state: str | None
	pos: tuple[int, int]


@dataclass
class ReplayData:
	scenario: str
	run: str
	terrain: np.ndarray
	agent_records: pd.DataFrame
	model_records: pd.DataFrame
	event_records: pd.DataFrame
	steps: list[int] = field(default_factory=list)

	@property
	def max_step(self) -> int:
		return max(self.steps) if self.steps else 0

	def agents_at(self, step: int) -> list[ReplayAgent]:
		if self.agent_records.empty:
			return []
		rows = self.agent_records[self.agent_records["step"] == step]
		agents: list[ReplayAgent] = []
		for row in rows.itertuples():
			if pd.isna(row.pos_x) or pd.isna(row.pos_y):
				continue
			agents.append(
				ReplayAgent(
					team=row.team,
					agent_class=row.agent_class,
					hp=float(row.hp),
					max_hp=float(row.max_hp),
					ai_state=row.ai_state,
					pos=(int(row.pos_x), int(row.pos_y)),
				)
			)
		return agents

	def model_at(self, step: int) -> dict | None:
		if self.model_records.empty:
			return None
		rows = self.model_records[self.model_records["step"] == step]
		return rows.iloc[0].to_dict() if len(rows) else None


def list_replay_scenarios(results_dir: Path = RESULTS_DIR) -> list[str]:
	if not results_dir.is_dir():
		return []
	return [
		child.name
		for child in sorted(results_dir.iterdir())
		if child.is_dir() and any(child.glob("run_*"))
	]


def list_runs(scenario: str, results_dir: Path = RESULTS_DIR) -> list[str]:
	scenario_dir = results_dir / scenario
	if not scenario_dir.is_dir():
		return []
	return sorted(path.name for path in scenario_dir.glob("run_*") if path.is_dir())


def _read_csv(path: Path) -> pd.DataFrame:
	if path.is_file():
		return pd.read_csv(path)
	return pd.DataFrame()


def _load_terrain(scenario: str, run_dir: Path) -> np.ndarray | None:
	map_file = run_dir / "map.csv"
	if map_file.is_file():
		return np.loadtxt(map_file, delimiter=",", dtype=int)
	try:
		scenario_def = load_scenario(scenario)
		return load_board(scenario_def["map"], base_dir=PROJECT_ROOT)
	except (ValueError, OSError, KeyError):
		return None


def load_run(
	scenario: str, run: str, results_dir: Path = RESULTS_DIR
) -> ReplayData | None:
	run_dir = results_dir / scenario / run
	if not run_dir.is_dir():
		return None

	agent_records = _read_csv(run_dir / "agent_telemetry.csv")
	model_records = _read_csv(run_dir / "model_telemetry.csv")
	event_records = _read_csv(run_dir / "event_telemetry.csv")
	if agent_records.empty and model_records.empty:
		return None

	terrain = _load_terrain(scenario, run_dir)
	if terrain is None:
		return None

	step_source = agent_records if not agent_records.empty else model_records
	steps = sorted(int(step) for step in step_source["step"].unique().tolist())

	return ReplayData(
		scenario=scenario,
		run=run,
		terrain=terrain,
		agent_records=agent_records,
		model_records=model_records,
		event_records=event_records,
		steps=steps,
	)


def _next_run_dir(scenario_dir: Path) -> Path:
	existing = {path.name for path in scenario_dir.glob("run_*") if path.is_dir()}
	index = 0
	while f"run_{index:03d}" in existing:
		index += 1
	return scenario_dir / f"run_{index:03d}"


def record_run(model, label: str | None = None, results_dir: Path = RESULTS_DIR) -> Path:
	"""Export a live model's telemetry plus its map for later replay.

	Each call gets its own ``run_NNN`` directory, so recording twice (even with
	the same label) never overwrites a previous run.
	"""
	name = label or f"gui_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
	scenario_dir = results_dir / name
	scenario_dir.mkdir(parents=True, exist_ok=True)
	run_dir = _next_run_dir(scenario_dir)
	run_dir.mkdir()

	model.telemetry.export_csv(run_dir)
	model.telemetry.export_json(run_dir)

	terrain = getattr(model, "initial_terrain", model.terrain)
	np.savetxt(run_dir / "map.csv", terrain, delimiter=",", fmt="%d")

	return run_dir

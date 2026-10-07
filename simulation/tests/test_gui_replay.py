from pathlib import Path

import numpy as np
import pandas as pd

from simulation.gui.model_factory import SimulationModel
from simulation.gui.replay import (
	ReplayData,
	_load_terrain,
	list_replay_scenarios,
	list_runs,
	load_run,
	record_run,
)


def _make_model() -> SimulationModel:
	model = SimulationModel(
		map_name="chokepoints_20x20.csv",
		blue_units="MainBattleTank",
		red_units="ReconSquad",
		blue_count=1,
		red_count=1,
		seed="1",
	)
	for _ in range(5):
		if model.running:
			model.step()
	return model


def _record_run(tmp_path: Path) -> Path:
	return record_run(_make_model(), label="test_run", results_dir=tmp_path)


def test_record_run_creates_replayable_files(tmp_path: Path) -> None:
	_record_run(tmp_path)

	assert list_replay_scenarios(tmp_path) == ["test_run"]
	assert list_runs("test_run", tmp_path) == ["run_000"]
	assert (tmp_path / "test_run" / "run_000" / "map.csv").is_file()


def test_load_run_reconstructs_frames(tmp_path: Path) -> None:
	_record_run(tmp_path)

	replay = load_run("test_run", "run_000", tmp_path)

	assert replay is not None
	assert replay.terrain.shape == (20, 20)
	assert replay.max_step >= 1

	agents = replay.agents_at(0)
	assert len(agents) == 2
	assert sorted(agent.team for agent in agents) == ["Blue", "Red"]
	assert all(agent.agent_class for agent in agents)


def test_load_run_model_record(tmp_path: Path) -> None:
	_record_run(tmp_path)

	replay = load_run("test_run", "run_000", tmp_path)

	assert replay is not None
	record = replay.model_at(0)
	assert record is not None
	assert record["alive_blue"] == 1
	assert record["alive_red"] == 1


def test_load_run_missing_returns_none(tmp_path: Path) -> None:
	assert load_run("missing", "run_000", tmp_path) is None


def test_load_run_without_map_returns_none(tmp_path: Path) -> None:
	run_dir = tmp_path / "no_map" / "run_000"
	run_dir.mkdir(parents=True)
	(run_dir / "agent_telemetry.csv").write_text("step,unique_id\n0,1\n")

	assert load_run("no_map", "run_000", tmp_path) is None


def test_list_runs_missing_scenario(tmp_path: Path) -> None:
	assert list_runs("nope", tmp_path) == []


def test_list_replay_scenarios_empty_dir(tmp_path: Path) -> None:
	assert list_replay_scenarios(tmp_path) == []


def test_record_run_does_not_overwrite_existing_run(tmp_path: Path) -> None:
	model = _make_model()

	first = record_run(model, label="dup", results_dir=tmp_path)
	second = record_run(model, label="dup", results_dir=tmp_path)

	assert first != second
	assert list_runs("dup", tmp_path) == ["run_000", "run_001"]
	assert (first / "map.csv").is_file()
	assert (second / "map.csv").is_file()


def test_load_run_with_model_records_only(tmp_path: Path) -> None:
	run_dir = tmp_path / "model_only" / "run_000"
	run_dir.mkdir(parents=True)
	np.savetxt(run_dir / "map.csv", np.zeros((4, 4), dtype=int), delimiter=",", fmt="%d")
	(run_dir / "model_telemetry.csv").write_text(
		"step,alive_blue,alive_red\n0,2,2\n1,1,2\n"
	)

	replay = load_run("model_only", "run_000", tmp_path)

	assert replay is not None
	assert replay.steps == [0, 1]
	assert replay.agents_at(0) == []
	assert replay.model_at(1)["alive_blue"] == 1


def test_model_at_missing_step_returns_none(tmp_path: Path) -> None:
	_record_run(tmp_path)
	replay = load_run("test_run", "run_000", tmp_path)

	assert replay is not None
	assert replay.model_at(9999) is None


def test_load_terrain_returns_none_for_missing_map(tmp_path: Path, monkeypatch) -> None:
	run_dir = tmp_path / "broken" / "run_000"
	run_dir.mkdir(parents=True)
	monkeypatch.setattr(
		"simulation.gui.replay.load_scenario", lambda name: {"map": "missing.csv"}
	)

	assert _load_terrain("broken", run_dir) is None


def test_load_run_without_telemetry_returns_none(tmp_path: Path) -> None:
	run_dir = tmp_path / "empty" / "run_000"
	run_dir.mkdir(parents=True)
	np.savetxt(run_dir / "map.csv", np.zeros((4, 4), dtype=int), delimiter=",", fmt="%d")

	assert load_run("empty", "run_000", tmp_path) is None


def test_agents_at_skips_rows_without_position() -> None:
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
			},
			{
				"step": 0,
				"unique_id": 2,
				"agent_class": "InfantrySquad",
				"team": "Red",
				"pos_x": None,
				"pos_y": None,
				"hp": 0,
				"max_hp": 100,
				"ai_state": "advance",
			},
		]
	)
	replay = ReplayData(
		scenario="s",
		run="run_000",
		terrain=np.zeros((4, 4), dtype=int),
		agent_records=agent_records,
		model_records=pd.DataFrame(),
		event_records=pd.DataFrame(),
		steps=[0],
	)

	agents = replay.agents_at(0)

	assert len(agents) == 1
	assert agents[0].pos == (1, 1)

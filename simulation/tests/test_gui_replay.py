from pathlib import Path

from simulation.gui.model_factory import SimulationModel
from simulation.gui.replay import (
	list_replay_scenarios,
	list_runs,
	load_run,
	record_run,
)


def _record_run(tmp_path: Path) -> Path:
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
	return record_run(model, label="test_run", results_dir=tmp_path)


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


def test_list_replay_scenarios_empty_dir(tmp_path: Path) -> None:
	assert list_replay_scenarios(tmp_path) == []

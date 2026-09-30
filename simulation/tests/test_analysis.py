import pytest

from simulation.analysis import ScenarioExplorer


def test_scenario_explorer_reports_missing_results_dir(tmp_path) -> None:
	missing_results = tmp_path / "does_not_exist"
	output_dir = tmp_path / "analysis"

	with pytest.raises(FileNotFoundError, match=r"Run 'python run_scenarios\.py' first"):
		ScenarioExplorer(missing_results, output_dir)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "scenarios" / "results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "scenarios" / "analysis"


class ScenarioAnalyzer:
	def __init__(self, results_dir: str | Path, output_dir: str | Path):
		self.results_dir = Path(results_dir)
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.summaries: list[dict[str, Any]] = []
		self._load_data()

	def _load_data(self) -> None:
		summary_path = self.results_dir / "all_summaries.json"
		if not summary_path.exists():
			raise FileNotFoundError(
				f"Results file not found: {summary_path}\n"
				"Run 'python run_scenarios.py' first."
			)
		with open(summary_path) as f:
			self.summaries = json.load(f)
		self.summaries.sort(key=lambda s: s.get("_filename", s.get("scenario_name", "")))

	@property
	def scenario_count(self) -> int:
		return len(self.summaries)

	def plot_win_rates(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		blue_wins = [s["blue_win_rate"] * 100 for s in self.summaries]
		red_wins = [s["red_win_rate"] * 100 for s in self.summaries]

		fig, ax = plt.subplots(figsize=(14, 7))
		x = np.arange(len(names))
		width = 0.25

		bars_blue = ax.bar(
			x - width, blue_wins, width, label="Blue Win %", color="#2166ac"
		)
		bars_red = ax.bar(x, red_wins, width, label="Red Win %", color="#b2182b")

		ax.set_ylabel("Win Rate (%)")
		ax.set_title("Scenario Win Rates — Blue vs Red")
		ax.set_xticks(x)
		ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
		ax.set_ylim(0, 110)
		ax.legend(loc="upper right")
		ax.grid(axis="y", alpha=0.3)
		ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

		for bar in bars_blue:
			h = bar.get_height()
			if h > 0:
				ax.text(
					bar.get_x() + bar.get_width() / 2,
					h + 1,
					f"{h:.0f}%",
					ha="center",
					fontsize=7,
				)
		for bar in bars_red:
			h = bar.get_height()
			if h > 0:
				ax.text(
					bar.get_x() + bar.get_width() / 2,
					h + 1,
					f"{h:.0f}%",
					ha="center",
					fontsize=7,
				)

		plt.tight_layout()
		path = self.output_dir / "01_win_rates.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_damage_comparison(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		blue_dmg = [s["avg_blue_damage"] for s in self.summaries]
		red_dmg = [s["avg_red_damage"] for s in self.summaries]

		fig, ax = plt.subplots(figsize=(14, 7))
		x = np.arange(len(names))
		width = 0.35

		ax.bar(x - width / 2, blue_dmg, width, label="Blue Avg Damage", color="#4393c3")
		ax.bar(x + width / 2, red_dmg, width, label="Red Avg Damage", color="#f4a582")

		ax.set_ylabel("Average Damage Dealt")
		ax.set_title("Total Damage Dealt per Team — All Scenarios")
		ax.set_xticks(x)
		ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
		ax.legend()
		ax.grid(axis="y", alpha=0.3)

		plt.tight_layout()
		path = self.output_dir / "02_damage_comparison.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_hit_rates(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		blue_hr = []
		red_hr = []
		for s in self.summaries:
			bh = s["avg_blue_hits"] / max(1, s["avg_blue_shots"]) * 100
			rh = s["avg_red_hits"] / max(1, s["avg_red_shots"]) * 100
			blue_hr.append(bh)
			red_hr.append(rh)

		fig, ax = plt.subplots(figsize=(14, 6))
		x = np.arange(len(names))
		width = 0.35

		ax.bar(x - width / 2, blue_hr, width, label="Blue Hit Rate %", color="#4393c3")
		ax.bar(x + width / 2, red_hr, width, label="Red Hit Rate %", color="#f4a582")
		ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3, label="50% baseline")

		ax.set_ylabel("Hit Rate (%)")
		ax.set_title("Hit Rate Comparison — Blue vs Red")
		ax.set_xticks(x)
		ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
		ax.set_ylim(0, 100)
		ax.legend()
		ax.grid(axis="y", alpha=0.3)

		plt.tight_layout()
		path = self.output_dir / "03_hit_rates.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_kill_comparison(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		blue_k = [s["avg_blue_kills"] for s in self.summaries]
		red_k = [s["avg_red_kills"] for s in self.summaries]

		fig, ax = plt.subplots(figsize=(14, 6))
		x = np.arange(len(names))
		width = 0.35

		ax.bar(x - width / 2, blue_k, width, label="Blue Avg Kills", color="#2166ac")
		ax.bar(x + width / 2, red_k, width, label="Red Avg Kills", color="#b2182b")

		ax.set_ylabel("Average Kills")
		ax.set_title("Kill Comparison — Blue vs Red")
		ax.set_xticks(x)
		ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
		ax.legend()
		ax.grid(axis="y", alpha=0.3)

		plt.tight_layout()
		path = self.output_dir / "04_kill_comparison.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_battle_efficiency(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		blue_eff = []
		red_eff = []
		for s in self.summaries:
			be = s["avg_blue_damage"] / max(1, s["avg_steps_run"])
			re = s["avg_red_damage"] / max(1, s["avg_steps_run"])
			blue_eff.append(be)
			red_eff.append(re)

		fig, ax = plt.subplots(figsize=(14, 6))
		x = np.arange(len(names))
		width = 0.35

		ax.bar(x - width / 2, blue_eff, width, label="Blue DPS", color="#4393c3")
		ax.bar(x + width / 2, red_eff, width, label="Red DPS", color="#f4a582")

		ax.set_ylabel("Damage Per Step")
		ax.set_title("Combat Efficiency (Damage per Step)")
		ax.set_xticks(x)
		ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
		ax.legend()
		ax.grid(axis="y", alpha=0.3)

		plt.tight_layout()
		path = self.output_dir / "05_battle_efficiency.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_summary_matrix(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		n = len(names)
		metrics = [
			"blue_win_rate",
			"avg_blue_damage",
			"avg_red_damage",
			"avg_blue_kills",
			"avg_red_kills",
			"avg_steps_run",
		]

		fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 2.5 * len(metrics)))
		fig.suptitle("Multi-Metric Scenario Comparison", fontsize=14, fontweight="bold")

		metric_labels = {
			"blue_win_rate": "Blue Win Rate",
			"avg_blue_damage": "Avg Blue Damage",
			"avg_red_damage": "Avg Red Damage",
			"avg_blue_kills": "Avg Blue Kills",
			"avg_red_kills": "Avg Red Kills",
			"avg_steps_run": "Avg Battle Duration (steps)",
		}
		colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n))

		for ax_idx, metric in enumerate(metrics):
			ax = axes[ax_idx]
			values = [s[metric] for s in self.summaries]
			bars = ax.barh(names, values, color=colors)
			ax.set_xlabel(metric_labels[metric])
			ax.grid(axis="x", alpha=0.3)

			for bar, v in zip(bars, values):
				ax.text(
					bar.get_width() + max(values) * 0.01,
					bar.get_y() + bar.get_height() / 2,
					f"{v:.1f}" if isinstance(v, float) else str(v),
					fontsize=7,
					va="center",
				)

		plt.tight_layout()
		path = self.output_dir / "06_summary_matrix.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def generate_report(self) -> Path:
		report_path = self.output_dir / "analysis_report.txt"
		lines: list[str] = []

		lines.append("=" * 70)
		lines.append("BATTLEFIELD SIMULATOR — STATISTICAL OVERVIEW")
		lines.append("=" * 70)
		lines.append(f"Scenarios analyzed: {len(self.summaries)}")
		lines.append("")

		for s in self.summaries:
			name = s["scenario_name"]
			hyp = s.get("hypothesis", "")
			bwr = s["blue_win_rate"]
			rwr = s["red_win_rate"]
			dr = s["draw_rate"]
			bk = s["avg_blue_kills"]
			rk = s["avg_red_kills"]
			bd = s["avg_blue_damage"]
			rd = s["avg_red_damage"]
			steps = s["avg_steps_run"]
			runs = s["run_count"]
			total_steps = s["total_steps"]

			lines.append("-" * 70)
			lines.append(f"SCENARIO: {name}")
			lines.append(
				f"  Runs: {runs} x {total_steps} steps | Avg duration: {steps:.1f} steps"
			)
			lines.append(f"  Hypothesis: {hyp}")
			lines.append("")
			lines.append(
				f"  Blue Win Rate: {bwr:.0%}  |  Red Win Rate: {rwr:.0%}  |  Draws: {dr:.0%}"  # noqa
			)
			lines.append(f"  Blue Kills: {bk:.1f} avg  |  Red Kills: {rk:.1f} avg")
			lines.append(f"  Blue Damage: {bd:.0f} avg  |  Red Damage: {rd:.0f} avg")
			lines.append("")

		report_text = "\n".join(lines)
		with open(report_path, "w") as f:
			f.write(report_text)

		print(report_text)
		return report_path

	def _short_name(self, s: dict) -> str:
		fn = s.get("_filename", "")
		if fn and len(fn) < 40:
			return fn
		name = s["scenario_name"]
		if name.startswith("H"):
			parts = name.split(":", 1)
			label = parts[0].strip()
			if len(label) > 25:
				return label[:25] + "..."
			return label
		return name[:30] + ("..." if len(name) > 30 else "")

	def generate_all(self) -> dict[str, Path]:
		print("Generating analysis charts...\n")
		return {
			"win_rates": self.plot_win_rates(),
			"damage_comparison": self.plot_damage_comparison(),
			"hit_rates": self.plot_hit_rates(),
			"kill_comparison": self.plot_kill_comparison(),
			"battle_efficiency": self.plot_battle_efficiency(),
			"summary_matrix": self.plot_summary_matrix(),
			"report": self.generate_report(),
		}


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(
		description="Analyze BattleField Simulator scenario results."
	)
	parser.add_argument(
		"--results-dir",
		type=str,
		default=str(DEFAULT_RESULTS_DIR),
		help=f"Directory containing scenario results (default: {DEFAULT_RESULTS_DIR})",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(DEFAULT_OUTPUT_DIR),
		help=f"Output directory for charts and report (default: {DEFAULT_OUTPUT_DIR})",
	)
	args = parser.parse_args()

	analyzer = ScenarioAnalyzer(args.results_dir, args.output_dir)
	paths = analyzer.generate_all()

	print(f"\nAnalysis complete. Output files in {args.output_dir}/:")
	for name, path in paths.items():
		print(f"  {name}: {path}")


if __name__ == "__main__":
	main()

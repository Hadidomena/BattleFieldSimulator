from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

	def plot_kill_death_ratio(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		blue_kdr = [
			s["avg_blue_kills"] / max(0.5, s["avg_blue_eliminated"])
			for s in self.summaries
		]
		red_kdr = [
			s["avg_red_kills"] / max(0.5, s["avg_red_eliminated"])
			for s in self.summaries
		]

		fig, ax = plt.subplots(figsize=(14, 6))
		x = np.arange(len(names))
		width = 0.35

		ax.bar(x - width / 2, blue_kdr, width, label="Blue KDR", color="#2166ac")
		ax.bar(x + width / 2, red_kdr, width, label="Red KDR", color="#b2182b")
		ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

		ax.set_ylabel("Kills per Loss")
		ax.set_title("Kill/Death Ratio per Team — All Scenarios")
		ax.set_xticks(x)
		ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
		ax.legend()
		ax.grid(axis="y", alpha=0.3)

		plt.tight_layout()
		path = self.output_dir / "07_kill_death_ratio.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_damage_efficiency(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		blue_dpe = [
			s["avg_blue_damage"] / max(1, s["avg_blue_shots"]) for s in self.summaries
		]
		red_dpe = [
			s["avg_red_damage"] / max(1, s["avg_red_shots"]) for s in self.summaries
		]

		fig, ax = plt.subplots(figsize=(14, 6))
		x = np.arange(len(names))
		width = 0.35

		ax.bar(x - width / 2, blue_dpe, width, label="Blue Dmg/Shot", color="#4393c3")
		ax.bar(x + width / 2, red_dpe, width, label="Red Dmg/Shot", color="#f4a582")

		ax.set_ylabel("Damage per Shot Fired")
		ax.set_title("Damage Efficiency (Damage per Shot) — All Scenarios")
		ax.set_xticks(x)
		ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
		ax.legend()
		ax.grid(axis="y", alpha=0.3)

		plt.tight_layout()
		path = self.output_dir / "08_damage_efficiency.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_battle_duration(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		avg_dur = [s["avg_steps_run"] for s in self.summaries]
		min_dur = [s["min_steps_run"] for s in self.summaries]
		max_dur = [s["max_steps_run"] for s in self.summaries]

		fig, ax = plt.subplots(figsize=(14, 6))
		x = np.arange(len(names))
		width = 0.6

		ax.bar(
			x,
			avg_dur,
			width,
			color="#7570b3",
			yerr=[
				[avg_dur[i] - min_dur[i] for i in range(len(names))],
				[max_dur[i] - avg_dur[i] for i in range(len(names))],
			],
			capsize=4,
			label="Avg duration",
		)

		total_steps = [s["total_steps"] for s in self.summaries]
		ax.scatter(
			x, total_steps, marker="x", color="red", zorder=5, label="Max steps cap"
		)

		ax.set_ylabel("Simulation Steps")
		ax.set_title("Battle Duration per Scenario (avg ± range)")
		ax.set_xticks(x)
		ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
		ax.legend()
		ax.grid(axis="y", alpha=0.3)

		plt.tight_layout()
		path = self.output_dir / "09_battle_duration.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_survival_rates(self) -> Path:
		names = [self._short_name(s) for s in self.summaries]
		blue_surv = []
		red_surv = []
		for s in self.summaries:
			blue_total = s["avg_blue_alive"] + s["avg_blue_eliminated"]
			red_total = s["avg_red_alive"] + s["avg_red_eliminated"]
			blue_surv.append(s["avg_blue_alive"] / max(1, blue_total) * 100)
			red_surv.append(s["avg_red_alive"] / max(1, red_total) * 100)

		fig, ax = plt.subplots(figsize=(14, 6))
		x = np.arange(len(names))
		width = 0.35

		ax.bar(
			x - width / 2, blue_surv, width, label="Blue Survivors %", color="#4393c3"
		)
		ax.bar(x + width / 2, red_surv, width, label="Red Survivors %", color="#f4a582")

		ax.set_ylabel("Survivors (% of starting force)")
		ax.set_title("Force Survival Rate per Scenario")
		ax.set_xticks(x)
		ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
		ax.set_ylim(0, 110)
		ax.legend()
		ax.grid(axis="y", alpha=0.3)

		plt.tight_layout()
		path = self.output_dir / "10_survival_rates.png"
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
			"kill_death_ratio": self.plot_kill_death_ratio(),
			"damage_efficiency": self.plot_damage_efficiency(),
			"battle_duration": self.plot_battle_duration(),
			"survival_rates": self.plot_survival_rates(),
			"report": self.generate_report(),
		}


class ScenarioExplorer:
	def __init__(self, results_dir: str | Path, output_dir: str | Path):
		self.results_dir = Path(results_dir)
		self.output_dir = Path(output_dir)
		self._scenario_dirs: dict[str, Path] = {}
		self._summary_map: dict[str, dict[str, Any]] = {}
		self._find_scenarios()

	def _find_scenarios(self) -> None:
		if not self.results_dir.exists():
			raise FileNotFoundError(
				f"Results directory not found: {self.results_dir}\n"
				"Run 'python run_scenarios.py' first."
			)

		summary_path = self.results_dir / "all_summaries.json"
		if summary_path.exists():
			with open(summary_path) as f:
				for s in json.load(f):
					fn = s.get("_filename", "")
					if fn:
						self._summary_map[fn] = s

		for d in sorted(self.results_dir.iterdir()):
			if d.is_dir() and d.name.startswith("scenario_"):
				name = d.name
				if name in self._summary_map:
					label = self._summary_map[name].get("scenario_name", name)
				else:
					summary_file = d / "summary.json"
					if summary_file.exists():
						with open(summary_file) as f:
							label = json.load(f).get("scenario_name", name)
					else:
						label = name
				self._scenario_dirs[label] = d

	@property
	def scenario_names(self) -> list[str]:
		return sorted(
			self._scenario_dirs.keys(),
			key=lambda n: (
				list(self._scenario_dirs.keys()).index(n)
				if n in self._scenario_dirs
				else 0
			),
		)

	def list_scenarios(self) -> list[str]:
		return list(self._scenario_dirs.keys())

	def _load_model_telemetry(self, scenario_name: str) -> pd.DataFrame:
		scenario_dir = self._scenario_dirs[scenario_name]
		frames = []
		for run_dir in sorted(scenario_dir.glob("run_*")):
			csv_path = run_dir / "model_telemetry.csv"
			if csv_path.exists():
				df = pd.read_csv(csv_path)
				df = df.assign(run=int(run_dir.name.split("_")[1]))
				frames.append(df)
		if not frames:
			raise FileNotFoundError(f"No model telemetry found for {scenario_name}")
		return pd.concat(frames, ignore_index=True)

	def _load_event_telemetry(self, scenario_name: str) -> pd.DataFrame:
		scenario_dir = self._scenario_dirs[scenario_name]
		frames = []
		for run_dir in sorted(scenario_dir.glob("run_*")):
			csv_path = run_dir / "event_telemetry.csv"
			if csv_path.exists():
				df = pd.read_csv(csv_path)
				df = df.assign(run=int(run_dir.name.split("_")[1]))
				frames.append(df)
		if not frames:
			raise FileNotFoundError(f"No event telemetry found for {scenario_name}")
		return pd.concat(frames, ignore_index=True)

	def _load_agent_telemetry(self, scenario_name: str) -> pd.DataFrame:
		scenario_dir = self._scenario_dirs[scenario_name]
		frames = []
		for run_dir in sorted(scenario_dir.glob("run_*")):
			csv_path = run_dir / "agent_telemetry.csv"
			if csv_path.exists():
				df = pd.read_csv(csv_path)
				df = df.assign(run=int(run_dir.name.split("_")[1]))
				frames.append(df)
		if not frames:
			raise FileNotFoundError(f"No agent telemetry found for {scenario_name}")
		return pd.concat(frames, ignore_index=True)

	def _scenario_output_dir(self, scenario_name: str) -> Path:
		path = self.output_dir / _sanitize_filename(scenario_name)
		path.mkdir(parents=True, exist_ok=True)
		return path

	def plot_population_curves(self, scenario_name: str) -> Path:
		df = self._load_model_telemetry(scenario_name)
		agg = (
			df.groupby("step")[["alive_blue", "alive_red"]]
			.agg(
				mean_blue=("alive_blue", "mean"),
				min_blue=("alive_blue", "min"),
				max_blue=("alive_blue", "max"),
				mean_red=("alive_red", "mean"),
				min_red=("alive_red", "min"),
				max_red=("alive_red", "max"),
			)
			.reset_index()
		)

		fig, ax = plt.subplots(figsize=(12, 6))
		ax.fill_between(
			agg["step"],
			agg["min_blue"],
			agg["max_blue"],
			alpha=0.2,
			color="#2166ac",
		)
		ax.plot(
			agg["step"], agg["mean_blue"], color="#2166ac", linewidth=2, label="Blue"
		)
		ax.fill_between(
			agg["step"],
			agg["min_red"],
			agg["max_red"],
			alpha=0.2,
			color="#b2182b",
		)
		ax.plot(agg["step"], agg["mean_red"], color="#b2182b", linewidth=2, label="Red")

		ax.set_xlabel("Step")
		ax.set_ylabel("Agents Alive")
		ax.set_title(f"Population Over Time — {scenario_name}")
		ax.legend()
		ax.grid(alpha=0.3)
		ax.set_ylim(bottom=0)
		ax.set_xlim(left=0)

		plt.tight_layout()
		path = self._scenario_output_dir(scenario_name) / "population_curves.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_damage_accumulation(self, scenario_name: str) -> Path:
		df = self._load_model_telemetry(scenario_name)
		agg = (
			df.groupby("step")[["damage_blue", "damage_red"]]
			.agg(
				mean_blue=("damage_blue", "mean"),
				min_blue=("damage_blue", "min"),
				max_blue=("damage_blue", "max"),
				mean_red=("damage_red", "mean"),
				min_red=("damage_red", "min"),
				max_red=("damage_red", "max"),
			)
			.reset_index()
		)

		fig, ax = plt.subplots(figsize=(12, 6))
		ax.fill_between(
			agg["step"],
			agg["min_blue"],
			agg["max_blue"],
			alpha=0.2,
			color="#4393c3",
		)
		ax.plot(
			agg["step"], agg["mean_blue"], color="#4393c3", linewidth=2, label="Blue"
		)
		ax.fill_between(
			agg["step"],
			agg["min_red"],
			agg["max_red"],
			alpha=0.2,
			color="#f4a582",
		)
		ax.plot(agg["step"], agg["mean_red"], color="#f4a582", linewidth=2, label="Red")

		ax.set_xlabel("Step")
		ax.set_ylabel("Cumulative Damage Dealt")
		ax.set_title(f"Damage Accumulation Over Time — {scenario_name}")
		ax.legend()
		ax.grid(alpha=0.3)
		ax.set_xlim(left=0)

		plt.tight_layout()
		path = self._scenario_output_dir(scenario_name) / "damage_accumulation.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_ai_state_distribution(self, scenario_name: str) -> Path:
		df = self._load_agent_telemetry(scenario_name)
		state_order = ["advance", "engage", "retreat", "patrol"]
		state_colors = {
			"advance": "#4393c3",
			"engage": "#b2182b",
			"retreat": "#f4a582",
			"patrol": "#999999",
		}

		state_counts = df.groupby(["run", "ai_state"]).size().unstack(fill_value=0)
		for s in state_order:
			if s not in state_counts.columns:
				state_counts[s] = 0
		state_counts = state_counts[state_order]
		state_pct = state_counts.div(state_counts.sum(axis=1), axis=0) * 100

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

		means = state_pct.mean()
		colors = [state_colors[s] for s in state_order]
		ax1.pie(
			means.values,
			labels=[f"{s}\n({v:.0f}%)" for s, v in zip(state_order, means.values)],
			colors=colors,
			startangle=90,
		)
		ax1.set_title(f"AI State Distribution — {scenario_name}")

		for run_idx in range(len(state_pct)):
			bottom = 0
			for s in state_order:
				val = state_pct.iloc[run_idx][s]
				if val > 0:
					ax2.bar(
						run_idx, val, bottom=bottom, color=state_colors[s], width=0.8
					)
					bottom += val
		ax2.set_xlabel("Run")
		ax2.set_ylabel("Percentage of Agent-Steps")
		ax2.set_title("Per-Run AI State Breakdown")
		ax2.set_ylim(0, 100)
		ax2.legend(
			[
				mpatches.Rectangle((0, 0), 1, 1, color=state_colors[s])
				for s in state_order
			],
			state_order,
			loc="upper right",
		)

		plt.tight_layout()
		path = self._scenario_output_dir(scenario_name) / "ai_state_distribution.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_event_timeline(self, scenario_name: str) -> Path:
		df = self._load_event_telemetry(scenario_name)
		hits = df[df["event_type"] == "hit"].copy()
		misses = df[df["event_type"] == "miss"].copy()
		eliminations = df[df["event_type"] == "elimination"].copy()

		fig, ax = plt.subplots(figsize=(14, 5))

		if len(misses):
			ax.scatter(
				misses["step"],
				misses["run"],
				marker="x",
				color="#cccccc",
				alpha=0.3,
				s=10,
				label="Miss",
			)
		if len(hits):
			ax.scatter(
				hits["step"],
				hits["run"],
				c=hits["damage"],
				cmap="RdYlGn_r",
				s=20,
				edgecolors="none",
				label="Hit",
			)
		if len(eliminations):
			ax.scatter(
				eliminations["step"],
				eliminations["run"],
				marker="D",
				color="black",
				s=30,
				label="Elimination",
			)

		ax.set_xlabel("Step")
		ax.set_ylabel("Run")
		ax.set_title(f"Event Timeline — {scenario_name}")
		if len(hits):
			cbar = plt.colorbar(
				ax.collections[1] if len(misses) else ax.collections[0], ax=ax
			)
			cbar.set_label("Damage")
		ax.legend(loc="upper right")
		ax.grid(alpha=0.2)

		plt.tight_layout()
		path = self._scenario_output_dir(scenario_name) / "event_timeline.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_damage_histogram(self, scenario_name: str) -> Path:
		df = self._load_event_telemetry(scenario_name)
		hit_damages = df.loc[df["event_type"] == "hit", "damage"]

		fig, ax = plt.subplots(figsize=(10, 5))
		ax.hist(hit_damages, bins=30, color="#4393c3", edgecolor="white", alpha=0.8)
		ax.axvline(
			hit_damages.mean(),
			color="#b2182b",
			linestyle="--",
			linewidth=2,
			label=f"Mean: {hit_damages.mean():.1f}",
		)
		ax.axvline(
			hit_damages.median(),
			color="black",
			linestyle=":",
			linewidth=2,
			label=f"Median: {hit_damages.median():.0f}",
		)

		ax.set_xlabel("Damage per Hit")
		ax.set_ylabel("Frequency")
		ax.set_title(f"Damage per Hit Distribution — {scenario_name}")
		ax.legend()
		ax.grid(axis="y", alpha=0.3)

		plt.tight_layout()
		path = self._scenario_output_dir(scenario_name) / "damage_histogram.png"
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def plot_hp_trajectories(self, scenario_name: str, run_index: int = 0) -> Path:
		df = self._load_agent_telemetry(scenario_name)
		run_df = df[df["run"] == run_index].copy()
		if run_df.empty:
			raise ValueError(f"Run {run_index} not found for {scenario_name}")

		fig, ax = plt.subplots(figsize=(12, 5))
		team_colors = {"Blue": "#2166ac", "Red": "#b2182b"}

		for agent_id in sorted(run_df["unique_id"].unique()):
			agent_df = run_df[run_df["unique_id"] == agent_id].sort_values("step")
			team = agent_df["team"].iloc[0]
			cls = agent_df["agent_class"].iloc[0]
			ax.plot(
				agent_df["step"],
				agent_df["hp"],
				color=team_colors.get(team, "gray"),
				alpha=0.8,
				linewidth=1.5,
				label=f"{team} {cls} #{agent_id}"
				if agent_id == run_df["unique_id"].unique().min() or True
				else "",
			)
			ax.scatter(
				agent_df["step"].iloc[-1:],
				agent_df["hp"].iloc[-1:],
				color=team_colors.get(team, "gray"),
				s=30,
				zorder=5,
			)

		ax.set_xlabel("Step")
		ax.set_ylabel("HP")
		ax.set_title(f"Agent HP Trajectories — {scenario_name} (Run {run_index})")
		ax.set_ylim(bottom=0)
		ax.legend(fontsize=7, ncol=2)
		ax.grid(alpha=0.3)

		plt.tight_layout()
		name_slug = f"run_{run_index:03d}"
		path = (
			self._scenario_output_dir(scenario_name) / f"hp_trajectories_{name_slug}.png"
		)
		fig.savefig(path, dpi=150)
		plt.close(fig)
		return path

	def explore_scenario(
		self, scenario_name: str, sample_run: int | None = 0
	) -> dict[str, Path]:
		paths: dict[str, Path] = {}

		print(f"\n  Exploring: {scenario_name}")
		for plot_name, method in [
			("population_curves", self.plot_population_curves),
			("damage_accumulation", self.plot_damage_accumulation),
			("ai_state_distribution", self.plot_ai_state_distribution),
			("event_timeline", self.plot_event_timeline),
			("damage_histogram", self.plot_damage_histogram),
		]:
			try:
				p = method(scenario_name)
				paths[plot_name] = p
				print(f"    {plot_name}: {p.name}")
			except Exception as e:
				print(f"    {plot_name}: SKIPPED ({e})")

		if sample_run is not None:
			try:
				p = self.plot_hp_trajectories(scenario_name, sample_run)
				paths["hp_trajectories"] = p
				print(f"    hp_trajectories: {p.name}")
			except Exception as e:
				print(f"    hp_trajectories: SKIPPED ({e})")

		return paths

	def explore_all(
		self, filters: list[str] | None = None
	) -> dict[str, dict[str, Path]]:
		results: dict[str, dict[str, Path]] = {}
		names = self.scenario_names
		if filters:
			names = [n for n in names if any(f.lower() in n.lower() for f in filters)]
		for name in names:
			results[name] = self.explore_scenario(name)
		return results


def _sanitize_filename(name: str) -> str:
	return (
		"".join(c if c.isalnum() or c in " _-" else "_" for c in name)
		.strip()
		.replace(" ", "_")[:100]
	)


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
	parser.add_argument(
		"--explore",
		type=str,
		default=None,
		action="append",
		help="Generate per-scenario telemetry deep-dive plots. "
		"Value is a scenario name filter (repeatable). "
		"Omit value to explore all scenarios.",
		nargs="?",
	)
	parser.add_argument(
		"--no-aggregate",
		action="store_true",
		help="Skip aggregate cross-scenario charts.",
	)
	parser.add_argument(
		"--list-scenarios",
		action="store_true",
		help="List available scenarios for exploration and exit.",
	)
	parser.add_argument(
		"--sample-run",
		type=int,
		default=0,
		help="Run index for per-agent HP trajectory plots (default: 0).",
	)
	args = parser.parse_args()

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	paths: dict[str, Path] = {}

	if args.list_scenarios:
		explorer = ScenarioExplorer(args.results_dir, output_dir)
		names = explorer.list_scenarios()
		print(f"Available scenarios ({len(names)}):")
		for n in names:
			print(f"  {n}")
		return

	if not args.no_aggregate:
		analyzer = ScenarioAnalyzer(args.results_dir, output_dir)
		paths.update(analyzer.generate_all())
		print()

	if args.explore is not None:
		filters = [f for f in args.explore if f]
		explorer = ScenarioExplorer(args.results_dir, output_dir)
		print("Exploring scenarios...")
		explorer.explore_all(filters=filters)

	print(f"\nOutput files in {output_dir}/:")
	for name, path in paths.items():
		print(f"  {name}: {path}")


if __name__ == "__main__":
	main()

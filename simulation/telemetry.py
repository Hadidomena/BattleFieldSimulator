from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from simulation.model import BattlefieldModel


class TelemetryCollector:
	def __init__(self) -> None:
		self.model_records: list[dict] = []
		self.agent_records: list[dict] = []
		self.event_records: list[dict] = []

	def record_step(self, model: BattlefieldModel) -> None:
		step = model.steps
		alive_blue = sum(
			1
			for a in model.agents
			if getattr(a, "team", None) == "Blue" and getattr(a, "hp", 0) > 0
		)
		alive_red = sum(
			1
			for a in model.agents
			if getattr(a, "team", None) == "Red" and getattr(a, "hp", 0) > 0
		)

		self.model_records.append(
			{
				"step": step,
				"alive_blue": alive_blue,
				"alive_red": alive_red,
				"eliminated_blue": model.eliminated_by_team["Blue"],
				"eliminated_red": model.eliminated_by_team["Red"],
				"kills_blue": model.kills_by_team["Blue"],
				"kills_red": model.kills_by_team["Red"],
				"shots_blue": model.shots_by_team["Blue"],
				"shots_red": model.shots_by_team["Red"],
				"hits_blue": model.hits_by_team["Blue"],
				"hits_red": model.hits_by_team["Red"],
				"damage_blue": model.damage_by_team["Blue"],
				"damage_red": model.damage_by_team["Red"],
				"battle_over": model.is_battle_over(),
			}
		)

		for agent in model.agents:
			agent_class = type(agent).__name__
			self.agent_records.append(
				{
					"step": step,
					"unique_id": agent.unique_id,
					"agent_class": agent_class,
					"team": getattr(agent, "team", None),
					"pos_x": agent.pos[0] if agent.pos else None,
					"pos_y": agent.pos[1] if agent.pos else None,
					"hp": getattr(agent, "hp", 0),
					"max_hp": getattr(agent, "max_hp", 0),
					"hp_ratio": getattr(agent, "hp", 0)
					/ max(1, getattr(agent, "max_hp", 1)),
					"ai_state": getattr(agent, "ai_state", None),
					"firepower": getattr(agent, "firepower", 0),
					"attack_range": getattr(agent, "attack_range", 0.0),
					"observation_range": getattr(agent, "observation_range", 0),
					"mobility": getattr(agent, "mobility", 0),
					"armor": getattr(agent, "armor", 0.0),
					"accuracy": getattr(agent, "accuracy", 0.0),
					"last_damage_dealt": getattr(agent, "last_damage_dealt", 0),
					"last_damage_taken": getattr(agent, "last_damage_taken", 0),
					"visible_enemies_count": len(
						getattr(agent, "last_detection_scores", {})
					),
					"is_alive": getattr(agent, "hp", 0) > 0,
				}
			)

	def record_shot(
		self,
		step: int,
		attacker_id: int,
		attacker_team: str,
		defender_id: int,
		defender_team: str,
		hit: bool,
		damage: int,
	) -> None:
		self.event_records.append(
			{
				"step": step,
				"event_type": "hit" if hit else "miss",
				"actor_id": attacker_id,
				"actor_team": attacker_team,
				"target_id": defender_id,
				"target_team": defender_team,
				"damage": damage if hit else 0,
			}
		)

	def record_elimination(
		self,
		step: int,
		eliminated_id: int,
		eliminated_team: str,
		killer_id: int | None,
		killer_team: str | None,
	) -> None:
		self.event_records.append(
			{
				"step": step,
				"event_type": "elimination",
				"actor_id": killer_id,
				"actor_team": killer_team,
				"target_id": eliminated_id,
				"target_team": eliminated_team,
				"damage": None,
			}
		)

	def export_csv(self, output_dir: str | Path) -> list[Path]:
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

		files: list[Path] = []
		model_path = output_dir / "model_telemetry.csv"
		agent_path = output_dir / "agent_telemetry.csv"
		event_path = output_dir / "event_telemetry.csv"

		self._export_records_csv(self.model_records, model_path)
		files.append(model_path)
		if self.agent_records:
			self._export_records_csv(self.agent_records, agent_path)
			files.append(agent_path)
		if self.event_records:
			self._export_records_csv(self.event_records, event_path)
			files.append(event_path)

		return files

	def export_json(self, output_dir: str | Path) -> Path:
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

		filepath = output_dir / "telemetry.json"
		data = {
			"model_telemetry": self.model_records,
			"agent_telemetry": self.agent_records,
			"event_telemetry": self.event_records,
		}
		with open(filepath, "w") as f:
			json.dump(data, f, indent="\t")

		return filepath

	@staticmethod
	def _export_records_csv(records: list[dict], filepath: Path) -> None:
		if not records:
			return
		fieldnames = list(records[0].keys())
		with open(filepath, "w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(records)

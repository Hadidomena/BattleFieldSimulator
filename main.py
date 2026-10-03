import argparse
import sys

import numpy as np

from simulation.model import BattlefieldModel
from simulation.utils import is_walkable, load_board, run_model

DEFAULT_STEPS = 10


def _default_spawn_points(
	board: np.ndarray, team: str, count: int
) -> list[tuple[int, int]]:
	width = board.shape[1]
	height = board.shape[0]
	positions: list[tuple[int, int]] = []
	if count <= 0:
		return positions

	if team == "Blue":
		candidate_rows = [1, 2, 3]
		candidate_columns = list(range(1, min(width - 1, count + 2)))
	else:
		candidate_rows = [height - 2, height - 3, height - 4]
		candidate_columns = list(range(max(1, width - count - 2), width - 1))

	for row in candidate_rows:
		for column in candidate_columns:
			if len(positions) >= count:
				return positions
			if is_walkable((column, row), board):
				positions.append((column, row))

	return positions


def _resolve_step_count(steps: int) -> int:
	if steps >= 0:
		return steps

	if sys.stdin.isatty():
		while True:
			try:
				value = int(input("How many steps should the simulation run for? "))
			except ValueError:
				print("Please enter a whole number.")
				continue
			except EOFError:
				break
			if value >= 0:
				return value
			print("Please enter a non-negative whole number.")

	print(f"No step count provided; defaulting to {DEFAULT_STEPS} steps.")
	return DEFAULT_STEPS


def main() -> None:
	parser = argparse.ArgumentParser(
		description=("Battlefield simulation - minimal skeleton"),
	)

	parser.add_argument(
		"--map",
		"-m",
		type=str,
		default=None,
		help=("Path to the board file (CSV or whitespace delimited)"),
	)

	parser.add_argument(
		"--steps",
		"-s",
		type=int,
		default=-1,
		help=("Number of steps to run the simulation for"),
	)
	parser.add_argument(
		"--blue-units",
		type=int,
		default=1,
		help=("Number of Blue units to place on the board"),
	)
	parser.add_argument(
		"--red-units",
		type=int,
		default=1,
		help=("Number of Red units to place on the board"),
	)
	parser.add_argument(
		"--output-dir",
		"-o",
		type=str,
		default=".",
		help=("Directory for telemetry output files (default: current directory)"),
	)
	parser.add_argument(
		"--export-format",
		type=str,
		default="both",
		choices=["csv", "json", "both"],
		help=("Export format for telemetry data (default: both)"),
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help=("Seed for the random number generator (default: random)"),
	)
	args = parser.parse_args()

	if args.map:
		board = load_board(args.map)
	else:
		board = np.zeros((10, 10), dtype=int)
		board[2:4, 2:4] = 1

	print(f"Loaded board (shape={board.shape}):")
	print(board)

	print("\n--- Initializing Environment Model ---")
	blue_spawn_points = _default_spawn_points(board, "Blue", args.blue_units)
	red_spawn_points = _default_spawn_points(board, "Red", args.red_units)
	model = BattlefieldModel(
		board,
		blue_spawn_points=blue_spawn_points,
		red_spawn_points=red_spawn_points,
		seed=args.seed,
	)

	num_steps = _resolve_step_count(args.steps)

	print(f"Running simulation for {num_steps} steps...\n")
	run_model(model, num_steps)

	print("\nResults (Analytics Module)")
	df = model.datacollector.get_model_vars_dataframe()
	print(df)

	print("\nTelemetry Export")
	if args.export_format in ("csv", "both"):
		files = model.telemetry.export_csv(args.output_dir)
		for f in files:
			print(f"Exported: {f}")
	if args.export_format in ("json", "both"):
		filepath = model.telemetry.export_json(args.output_dir)
		print(f"Exported: {filepath}")


if __name__ == "__main__":
	main()

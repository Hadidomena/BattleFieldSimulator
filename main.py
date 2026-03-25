import argparse
from pathlib import Path

import numpy as np

from simulation.model import BattlefieldModel


def load_board(path: str) -> np.ndarray:
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Map file not found: {path}")
	try:
		board = np.loadtxt(p, delimiter=",", dtype=int)
	except Exception:
		board = np.loadtxt(p, dtype=int)
	return board


def main() -> None:
	parser = argparse.ArgumentParser(
		description=("Symulacja pola walki — minimalny szkielet"),
	)

	parser.add_argument(
		"--map",
		"-m",
		type=str,
		default=None,
		help=("Ścieżka do pliku z planszą (CSV lub whitespace delimited)"),
	)

	args = parser.parse_args()

	if args.map:
		board = load_board(args.map)
	else:
		board = np.zeros((10, 10), dtype=int)
		board[2:4, 2:4] = 1

	print(f"Plansza wczytana (shape={board.shape}):")
	print(board)

	print("\n--- Inicjalizacja Modelu Środowiska ---")
	model = BattlefieldModel(board)

	# Uruchamiamy symulację na zadaną ilość tur (np. 3 tury)
	num_steps = 3
	print(f"Uruchamiam symulację na {num_steps} tury...\n")
	for i in range(num_steps):
		model.step()

	print("\n=== Wyniki (Moduł Analityczny) ===")
	df = model.datacollector.get_model_vars_dataframe()
	print(df)


if __name__ == "__main__":
	main()

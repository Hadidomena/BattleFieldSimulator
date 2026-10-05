from __future__ import annotations

import sys
from pathlib import Path

from solara.__main__ import cli

APP_PATH = Path(__file__).resolve().parent / "app.py"


def main() -> None:
	sys.argv = ["solara", "run", str(APP_PATH), *sys.argv[1:]]
	cli()


if __name__ == "__main__":
	main()

# Description of Project
This project was made as a part of my Engineering degree in Applied Computer Science.

# Features
Current simulation mechanics and tools include:
- Pathfinding (A* and Dijkstra) with terrain-aware movement costs.
- Line of Sight (LoS) detection using Bresenham's algorithm and vision cone checks.
- Finite-state tactical logic: patrol, advance, engage, and retreat.
- Tactical unit classes in meso scale (squads and vehicles): infantry, recon, mechanized infantry, main battle tank.
- Combat mechanics with hit probability, damage model, armor, and elimination rules.
- Automated code formatting (`ruff`, `pre-commit`) and testing (`pytest` with coverage metrics).

# Used Resources

## Language
This project was made with Python.

## Libraries
Following libraries were used:
- mesa
- networkx
- matplotlib
- numpy
- pandas
- seaborn

Additionally, development and testing tools include:
- pytest & pytest-cov
- ruff
- pre-commit
All of them have been added to the **requirements.txt**.

# How to use
To run the current simulation skeleton with an auto-generated 10x10 board containing some obstacles, simply execute:
```bash
python main.py
```

To run with fixed number of turns:
```bash
python main.py --steps 20
```

You can also load a custom board containing a matrix of 0s (empty) and 1s (obstacle walls) from a CSV or whitespace-delimited text file:
```bash
python main.py --map custom_board.csv
```

To run the automated tests and check the code coverage:
```bash
pytest -v --cov=simulation
```

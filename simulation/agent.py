import mesa

from simulation.utils import has_line_of_sight


class CombatAgent(mesa.Agent):
	def __init__(
		self,
		model: mesa.Model,
		team: str,
		hp: int = 100,
		firepower: int = 10,
		observation_range: int = 5,
		mobility: int = 2,
	) -> None:
		super().__init__(model)
		self.team = team
		self.hp = hp
		self.firepower = firepower
		self.observation_range = observation_range
		self.mobility = mobility

	def step(self) -> None:
		if self.hp <= 0:
			return

		self.move()

		enemies = self.get_visible_enemies()
		enemy_ids = [e.unique_id for e in enemies]

		current_step = self.model.steps
		print(
			f"[Turn {current_step}] Agent {self.unique_id} "
			f"({self.team}) in {self.pos} ready. HP: {self.hp}. "
			f"Visible enemies: {enemy_ids}"
		)

	def get_visible_enemies(self) -> list["CombatAgent"]:
		enemies = []
		for agent in self.model.agents:
			if (
				isinstance(agent, CombatAgent)
				and agent.team != self.team
				and agent.hp > 0
			):
				dx = self.pos[0] - agent.pos[0]
				dy = self.pos[1] - agent.pos[1]
				dist = (dx**2 + dy**2) ** 0.5

				if dist <= self.observation_range:
					if has_line_of_sight(self.pos, agent.pos, self.model.terrain):
						enemies.append(agent)
		return enemies

	def move(self) -> None:
		possible_steps = self.model.grid.get_neighborhood(
			self.pos, moore=True, include_center=False
		)
		if hasattr(self.model, "terrain"):
			valid_steps = [
				pos for pos in possible_steps if self.model.terrain[pos[1], pos[0]] == 0
			]
		else:
			valid_steps = possible_steps

		if valid_steps:
			new_position = self.random.choice(valid_steps)
			self.model.grid.move_agent(self, new_position)

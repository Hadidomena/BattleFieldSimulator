import mesa


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

		# TODO: In the future algorithms for movement and decision taking
		current_step = self.model.steps
		print(
			f"[Turn {current_step}] Agent {self.unique_id} "
			f"({self.team}) in {self.pos} ready. HP: {self.hp}"
		)

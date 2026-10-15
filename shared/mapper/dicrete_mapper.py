import random
from numpy import float32 , linspace
from numpy.typing import NDArray
from gymnasium import spaces as gym_spaces

class DiscreteActionMapper:
	"""
	A class to map discrete action indices to continuous action values.
	"""
	def __init__(self, action_space: gym_spaces.Box, num_actions: int = 9) -> None:
		"""
		Initialize the DiscreteActionMapper.

		Args:
			action_space: The action space to map from.
			num_actions: The number of discrete actions to map to.
		"""
		if action_space.shape != (1,):
			raise ValueError("Expected a one-dimensional Box action space with shape (1,).")
		if num_actions < 2:
			raise ValueError("num_actions must be at least 2.")
		
		self.actions: NDArray[float32] = linspace(
			action_space.low[0],
			action_space.high[0],
			num_actions,
			dtype=float32,
		).reshape(num_actions, 1)

	def __len__(self) -> int:
		return len(self.actions)

	def to_continuous(self, action_index: int) -> NDArray[float32]:
		if not 0 <= action_index < len(self):
			raise IndexError("Action index is out of range.")
		return self.actions[action_index].copy()

	def sample_random_action(self) -> tuple[int, NDArray[float32]]:
		"""
		Sample a random action from the discrete action space.

		Returns:
			A random continuous action corresponding to a discrete index.
		"""
		random_index = random.randint(0, len(self) - 1)
		return random_index, self.to_continuous(random_index)
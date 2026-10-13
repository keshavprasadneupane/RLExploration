from collections import defaultdict
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from pathlib import Path

class CartPoleQLearningAgent:
	def __init__(
		self,
		env: gym.Env,
		learning_rate: float = 0.1,
		discount_factor: float = 0.99,
		initial_epsilon: float = 1.0,
		min_epsilon: float = 0.01,
		epsilon_decay: float = 0.995,
	) -> None:
		action_space = env.action_space

		if not isinstance(action_space, Discrete):
			raise TypeError("Expected a discrete action space.")

		self.env = env
		self.action_space = action_space

		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.epsilon = initial_epsilon
		self.min_epsilon = min_epsilon
		self.epsilon_decay = epsilon_decay

		self.bins: tuple[np.ndarray, ...] = (
			np.linspace(-4.8, 4.8, 100),
			np.linspace(-4.0, 4.0, 100),
			np.linspace(-0.418, 0.418, 100),
			np.linspace(-4.0, 4.0, 100),
		)

		self.q_table: defaultdict[tuple[int, ...], np.ndarray] = (
			defaultdict(
				lambda: np.zeros(
					self.action_space.n,
					dtype=np.float64,
				)
			)
		)

		self.errors: list[float] = []

	def save_q_table(self, file_path: Path) -> None:
		q_table = np.empty((), dtype=object)
		q_table[()] = dict(self.q_table)
		np.save(file_path, q_table, allow_pickle=True)

	def load_q_table(self, file_path: Path) -> None:
		q_table_dict = np.load(file_path, allow_pickle=True).item()
		self.q_table = defaultdict(
			lambda: np.zeros(self.action_space.n, dtype=np.float64),
			q_table_dict,
		)

	def discretize_state(
		self,
		state: np.ndarray,
	) -> tuple[int, ...]:
		return tuple(
			int(np.digitize(value, bin_edges))
			for value, bin_edges in zip(state, self.bins)
		)

	def choose_action(
		self,
		state: tuple[int, ...],
	) -> int:
		if np.random.random() < self.epsilon:
			return int(self.action_space.sample())

		return int(np.argmax(self.q_table[state]))

	def decay_epsilon(self) -> None:
		self.epsilon = max(
			self.min_epsilon,
			self.epsilon * self.epsilon_decay,
		)

	def update_q_value(
		self,
		state: tuple[int, ...],
		action: int,
		reward: float,
		terminated: bool,
		next_state: tuple[int, ...],
	) -> None:	
		best_next_action_value = (not terminated) * float(
			np.max(self.q_table[next_state])
		)

		td_target = (
			reward
			+ self.discount_factor * best_next_action_value
		)

		td_delta = td_target - self.q_table[state][action]

		self.q_table[state][action] += (
			self.learning_rate * td_delta
		)

		self.errors.append(float(td_delta))
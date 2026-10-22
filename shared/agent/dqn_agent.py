import random
from collections import deque
from typing import Any, Self

import numpy as np
import torch

from shared.network.nnetwork import NNewtork


class ReplayBuffer:
	"""Stores and randomly samples experience transitions for DQN."""

	def __init__(self, capacity: int = 100_000) -> None:
		if capacity <= 0:
			raise ValueError("Replay buffer capacity must be positive.")

		self.capacity = capacity
		self._buffer: deque[
			tuple[np.ndarray, int, float, np.ndarray, bool]
		] = deque(maxlen=capacity)

	def __len__(self) -> int:
		return len(self._buffer)

	def add(
		self,
		state: np.ndarray,
		action: int,
		reward: float,
		next_state: np.ndarray,
		done: bool,
	) -> None:
		"""Adds one transition to replay memory."""
		self._buffer.append(
			(
				np.array(state, dtype=np.float32, copy=True),
				int(action),
				float(reward),
				np.array(next_state, dtype=np.float32, copy=True),
				bool(done),
			)
		)

	def sample(
		self,
		batch_size: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""Samples a random mini-batch without replacement."""
		if batch_size <= 0:
			raise ValueError("Batch size must be positive.")

		if len(self._buffer) < batch_size:
			raise ValueError("Not enough experiences to sample a batch.")

		batch = random.sample(self._buffer, batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)

		return (
			np.stack(states),
			np.asarray(actions, dtype=np.int64),
			np.asarray(rewards, dtype=np.float32),
			np.stack(next_states),
			np.asarray(dones, dtype=np.float32),
		)

	def get_save_state(self) -> dict[str, Any]:
		"""Returns replay memory in a checkpoint-friendly structure."""
		return {
			"capacity": self.capacity,
			"buffer": list(self._buffer),
		}

	def load_save_state(self, state: dict[str, Any]) -> None:
		"""Restores replay memory from a checkpoint."""
		capacity = int(state.get("capacity", self.capacity))
		if capacity <= 0:
			raise ValueError("Replay buffer capacity must be positive.")

		self.capacity = capacity
		self._buffer = deque(maxlen=capacity)

		for transition in state.get("buffer", []):
			self.add(*transition)


class DQNetworkAgent:
	"""
	Shared policy and target network functionality for DQN.

	Supports continuous observations and discrete actions.
	"""

	def __new__(cls, *args: Any, **kwargs: Any) -> "Self":
		if cls is DQNetworkAgent:
			raise TypeError(
				"DQNetworkAgent is a base class and cannot be instantiated directly."
			)
		return super().__new__(cls)

	def __init__(
		self,
		*,
		network: NNewtork,
		target_update_freq: int = 1000,
	) -> None:
		if target_update_freq <= 0:
			raise ValueError("Target update frequency must be positive.")

		self.network = network
		self.device = network.device
		self.target_update_freq = target_update_freq

		self.target_network = self.network.deep_copy()
		self.steps = 0

		self.update_target_network()

	def update_target_network(self) -> None:
		"""Synchronizes target network weights with the policy network."""
		self.target_network.load_state_dict(self.network.state_dict())
		self.target_network.eval()

	def get_action(self, state: np.ndarray) -> tuple[int, float]:
		"""Selects the greedy action and returns its estimated Q-value."""
		q_values = self.network.predict(state)
		action = int(np.argmax(q_values))
		return action, float(q_values[action])

	def _predict_target(self, state: np.ndarray) -> np.ndarray:
		"""Computes Q-values using the target network."""
		return self.target_network.predict(state)


class DQNetworkTrainAgent(DQNetworkAgent):
	"""DQN training agent with experience replay."""

	def __init__(
		self,
		*,
		network: NNewtork,
		optimizer: torch.optim.Optimizer,
		target_update_freq: int = 1000,
		replay_capacity: int = 100_000,
		batch_size: int = 64,
	) -> None:
		super().__init__(
			network=network,
			target_update_freq=target_update_freq,
		)

		if batch_size <= 0:
			raise ValueError("Batch size must be positive.")

		self.optimizer = optimizer
		self.learning_rate = self.optimizer.param_groups[0].get("lr", 0.001)

		self.batch_size = batch_size
		self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

	def update_weights(
		self,
		state: np.ndarray,
		action: int,
		reward: float,
		next_state: np.ndarray,
		done: bool,
		discount_factor: float = 0.99,
	) -> float:
		"""
		Stores a transition and trains on a random mini-batch.

		Returns the loss, or 0.0 while the replay buffer is filling.
		"""
		if not 0.0 <= discount_factor <= 1.0:
			raise ValueError("Discount factor must be between 0 and 1.")

		# Store the latest environment transition.
		self.replay_buffer.add(
			state=state,
			action=action,
			reward=reward,
			next_state=next_state,
			done=done,
		)

		# Wait until a complete mini-batch is available.
		if len(self.replay_buffer) < self.batch_size:
			return 0.0

		states, actions, rewards, next_states, dones = (
			self.replay_buffer.sample(self.batch_size)
		)

		states_tensor = torch.as_tensor(
			states, dtype=torch.float32, device=self.device
		)
		actions_tensor = torch.as_tensor(
			actions, dtype=torch.long, device=self.device
		)
		rewards_tensor = torch.as_tensor(
			rewards, dtype=torch.float32, device=self.device
		)
		next_states_tensor = torch.as_tensor(
			next_states, dtype=torch.float32, device=self.device
		)
		dones_tensor = torch.as_tensor(
			dones, dtype=torch.float32, device=self.device
		)

		self.network.train()

		# Select the Q-value corresponding to each sampled action.
		all_q_values: torch.Tensor = self.network(states_tensor)
		current_q_values = all_q_values.gather(
			dim=1,
			index=actions_tensor.unsqueeze(1),
		).squeeze(1)

		# Calculate target Q-values without tracking gradients.
		with torch.no_grad():
			next_q_values: torch.Tensor = self.target_network(
				next_states_tensor
			)
			max_next_q_values = next_q_values.max(dim=1).values

			target_q_values = rewards_tensor + (
				discount_factor
				* (1.0 - dones_tensor)
				* max_next_q_values
			)

		loss: torch.Tensor = self.network.loss(
			current_q_values,
			target_q_values,
		)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.steps += 1

		if self.steps % self.target_update_freq == 0:
			self.update_target_network()

		return float(loss.item())

	def get_save_state(self) -> dict[str, Any]:
		"""Returns all training states, including replay memory."""
		return {
			"network_state_dict": self.network.state_dict(),
			"target_network_state_dict": self.target_network.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"steps": self.steps,
			"batch_size": self.batch_size,
			"replay_buffer_state": self.replay_buffer.get_save_state(),
		}

	def load_save_state(self, state: dict[str, Any]) -> None:
		"""Restores training state, supporting older checkpoints."""
		self.network.load_state_dict(state["network_state_dict"])
		self.target_network.load_state_dict(
			state["target_network_state_dict"]
		)
		self.optimizer.load_state_dict(state["optimizer_state_dict"])
		self.steps = int(state["steps"])

		self.batch_size = int(state.get("batch_size", self.batch_size))

		if "replay_buffer_state" in state:
			self.replay_buffer.load_save_state(
				state["replay_buffer_state"]
			)


class DQNetworkTestAgent(DQNetworkAgent):
	"""DQN inference agent without replay memory or an optimizer."""

	def __init__(self, *, network: NNewtork) -> None:
		super().__init__(network=network)

	def load_save_state(self, state: dict[str, Any]) -> None:
		"""Loads model weights for inference."""
		self.network.load_state_dict(state["network_state_dict"])
		self.target_network.load_state_dict(
			state["target_network_state_dict"]
		)
		self.network.eval()
		self.target_network.eval()

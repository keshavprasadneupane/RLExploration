import numpy as np
import torch

from shared.network.nnetwork import NNewtork


class SimpleDQNAgentBase:
	"""
	Shared policy and target network functionality for DQN.
	
		Designed for environments with continuous observations and discrete actions.
		Observations are passed directly to the neural network without discretization,
		while the network outputs one Q-value per discrete action. The agent selects
		actions based on these Q-values, using an exploration strategy during training
		when applicable.
	"""

	def __new__(cls, *args, **kwargs):
		"""
		Prevent direct instantiation of the base class.
		This enforces abstract-like behavior without requiring ABC.
		Only subclasses should be instantiated.
		"""
		if cls is SimpleDQNAgentBase:
			raise TypeError(
				"SimpleDQNAgentBase is an abstract base class and cannot be instantiated directly."
			)
		return super().__new__(cls)
	
	def __init__(
		self,
		*,
		network: NNewtork,
		target_update_freq: int = 1000,
	) -> None:
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
		value = float(q_values[action])
		return action, value

	def _predict_target(self, state: np.ndarray) -> np.ndarray:
		"""Computes Q-values using the target network."""
		return self.target_network.predict(state)



class SimpleDQNTrainAgent(SimpleDQNAgentBase):
	"""DQN agent supporting training and checkpoint management."""

	def __init__(
		self,
		*,
		network: NNewtork,
		optimizer: torch.optim.Optimizer,
		target_update_freq: int = 1000,
	) -> None:
		super().__init__(
			network=network,
			target_update_freq=target_update_freq,
		)
		self.optimizer = optimizer
		self.learning_rate = self.optimizer.param_groups[0].get("lr", 0.001)

	def update_weights(
		self,
		state: np.ndarray,
		action: int,
		reward: float,
		next_state: np.ndarray,
		done: bool,
		discount_factor: float = 0.99,
	) -> float:
		"""Performs a single-transition DQN gradient update."""
		state_tensor = torch.as_tensor(
			state, dtype=torch.float32, device=self.device
		)
		next_state_tensor = torch.as_tensor(
			next_state, dtype=torch.float32, device=self.device
		)

		self.network.train()

		q_values: torch.Tensor = self.network(state_tensor)
		current_q: torch.Tensor = q_values[action]

		with torch.no_grad():
			if done:
				target_q = torch.tensor(
					reward, dtype=torch.float32, device=self.device
				)
			else:
				next_q_values: torch.Tensor = self.target_network(
					next_state_tensor
				)
				target_q = reward + discount_factor * next_q_values.max()

		loss: torch.Tensor = self.network.loss(current_q, target_q)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.steps += 1

		if self.steps % self.target_update_freq == 0:
			self.update_target_network()

		return loss.item()

	def get_save_state(self) -> dict:
		"""Returns the model, target network, optimizer, and step states."""
		return {
			"network_state_dict": self.network.state_dict(),
			"target_network_state_dict": self.target_network.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"steps": self.steps,
		}

	def load_save_state(self, state: dict) -> None:
		"""Restores the model, target network, optimizer, and step states."""
		self.network.load_state_dict(state["network_state_dict"])
		self.target_network.load_state_dict(
			state["target_network_state_dict"]
		)
		self.optimizer.load_state_dict(state["optimizer_state_dict"])
		self.steps = state["steps"]



class SimpleDQNTestAgent(SimpleDQNAgentBase):
	"""DQN agent for inference without an optimizer."""

	def __init__(self, *, network: NNewtork) -> None:
		super().__init__(network=network)

	def load_save_state(self, state: dict) -> None:
		"""Loads model weights for inference."""
		self.network.load_state_dict(state["network_state_dict"])
		self.target_network.load_state_dict(
			state["target_network_state_dict"]
		)


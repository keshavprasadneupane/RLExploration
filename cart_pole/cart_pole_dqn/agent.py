from typing import Callable
from zipfile import Path
import numpy as np

from torch import nn
import torch

from cart_pole.cart_pole_q_table import agent

from .network import Network



OptimizerFactory = Callable[..., torch.optim.Optimizer]
	

class CartPoleDQNAgent:
	"""Manage the policy network and target network for DQN."""

	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		hidden_dim: int = 8,
		target_update_freq: int = 1000,
		learning_rate: float = 0.001,
		optimizer: OptimizerFactory = torch.optim.Adam,
		criterion: nn.Module | None = None,
		device: str | torch.device = "cpu",
	) -> None:
		self.target_update_freq = target_update_freq
		self.learning_rate = learning_rate
		self.steps = 0
		
		self.device = torch.device(device)

		self.network = Network(
			input_dim, output_dim, hidden_dim, criterion
		).to(self.device)

		self.target_network = self.network.copy().to(self.device)

		self.optimizer = optimizer(
			self.network.parameters(),
			lr=learning_rate,
		)

		self.update_target_network()

	def update_target_network(self) -> None:
		"""Synchronize target network weights with the policy network."""
		self.target_network.load_state_dict(
			self.network.state_dict()
		)
		self.target_network.eval()

	def get_action(self, state: np.ndarray) -> tuple[int, float]:
		"""Return the greedy action and its estimated Q-value."""
		q_values = self.network.predict(state)
		action = int(np.argmax(q_values))
		value = float(q_values[action])

		return action, value

	def _predict_target(self, state: np.ndarray) -> np.ndarray:
		"""Predict Q-values using the target network."""
		return self.target_network.predict(state)

	def update_weights(
		self,
		state: np.ndarray,
		action: int,
		reward: float,
		next_state: np.ndarray,
		done: bool,
		discount_factor: float = 0.99,
	) -> float:
		"""Perform one DQN update using a single transition."""

		state_tensor = torch.as_tensor(
			state,
			dtype=torch.float32,
			device=self.device,
		)
		next_state_tensor = torch.as_tensor(
			next_state,
			dtype=torch.float32,
			device=self.device,
		)

		
		self.network.train()
		q_values = self.network(state_tensor)
		current_q = q_values[action]

	
		with torch.no_grad():
			if done:
				target_q = torch.tensor(
					reward,
					dtype=torch.float32,
					device=self.device,
				)
			else:
				next_q_values = self.target_network(next_state_tensor)
				target_q = (
					reward
					+ discount_factor * next_q_values.max()
				)

		
		loss = self.network.loss(current_q, target_q)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		
		self.steps += 1

		if self.steps % self.target_update_freq == 0:
			self.update_target_network()

		return loss.item()


	def get_save_state(self,) -> dict:
		return{
			"network_state_dict": self.network.state_dict(),
			"target_network_state_dict": (
				self.target_network.state_dict()
			),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"steps": self.steps,
		}

	def load_save_state(self, state: dict) -> None:
		self.network.load_state_dict(state["network_state_dict"])
		self.target_network.load_state_dict(
			state["target_network_state_dict"]
		)
		self.optimizer.load_state_dict(state["optimizer_state_dict"])
		self.steps = state["steps"]
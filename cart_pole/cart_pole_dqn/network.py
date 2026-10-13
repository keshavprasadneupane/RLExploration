
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Network(nn.Module):
	"""A feedforward neural network for estimating Q-values."""

	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		hidden_dim: int = 8,
		criterion: nn.Module | None = None,
	) -> None:
		super().__init__()

		self.nn = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
		)

		self.criterion = (
			criterion if criterion is not None
			else nn.SmoothL1Loss()
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.nn(x)

	def loss(
		self,
		predictions: torch.Tensor,
		targets: torch.Tensor,
	) -> torch.Tensor:
		"""Calculate the loss between predicted and target Q-values."""
		return self.criterion(predictions, targets)

	def predict(self, state: np.ndarray) -> np.ndarray:
		"""Predict Q-values without tracking gradients."""
		self.eval()

		with torch.no_grad():
			state_tensor = torch.as_tensor(
				state,
				dtype=torch.float32,
				device=next(self.parameters()).device,
			)
			q_values = self(state_tensor)

		return q_values.cpu().numpy()

	def copy(self) -> "Network":
		"""Return an independent copy of this network."""
		return copy.deepcopy(self)

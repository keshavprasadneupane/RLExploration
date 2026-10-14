import copy
import numpy as np
import torch
from torch import nn
from shared.validate import validate_device
from abc import ABC, abstractmethod
from typing import TypeVar


M = TypeVar("M", bound="NNewtork")

class NNewtork(nn.Module, ABC):
	"""
	A base neural network wrapper module.
	Manages a custom neural network module, loss function criterion, 
	and automatic device placement.
	"""
	
	def __init__(
		self,
		custom_nn: nn.Module,
		loss_function: nn.Module | None = None,
		device: str = "cpu",
	) -> None:
		"""Initializes the NNewtork instance.

		Args:
			custom_nn: The PyTorch neural network module to wrap.
			loss_function: Optional loss module. Defaults to SmoothL1Loss.
			device: The target computation device ("cpu" or "cuda").
		"""
		super().__init__()

		self.device = validate_device(device)
		self.nn = custom_nn.to(self.device)

		self.loss_function = (
			loss_function if loss_function is not None 
			else nn.SmoothL1Loss()
		)

	def loss(
		self,
		predictions: torch.Tensor,
		targets: torch.Tensor,
	) -> torch.Tensor:
		"""Computes the loss between prediction and target tensors."""
		return self.loss_function(predictions, targets)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Performs a forward pass through the underlying network."""
		return self.nn(x)

	def predict(self, state: np.ndarray) -> np.ndarray:
		"""Generates numpy-formatted predictions from a numpy array input without gradient tracking."""
		self.eval()
		with torch.no_grad():
			state_tensor = torch.as_tensor(
				state,
				dtype=torch.float32,
				device=next(self.parameters()).device,
			)
			q_values = self(state_tensor)

		return q_values.cpu().numpy()

	@abstractmethod
	def copy(self) -> "NNewtork":
		"""Returns a deep, independent copy of the network instance."""
		return self._copy(self)


	@staticmethod
	def _copy(newtork:M) -> M:
		"""Static method to create a deep copy of a given network."""
		return copy.deepcopy(newtork)
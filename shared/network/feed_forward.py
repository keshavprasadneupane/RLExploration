from torch import nn
from .nnetwork import NNewtork  




class FeedForward(NNewtork):
	"""
		A customizable feedforward neural network module.
	
		Supports dynamic hidden layer sizing, configurable activation functions, 
		custom loss criteria, and optional custom network overrides.
	
		Notes
		-----
		- Defaults to `nn.SmoothL1Loss` if no loss function is provided.
		- Device placement and tensor movement are managed automatically on forward/predict pass.
		- Optimization and training loops are intentionally decoupled and managed externally.
	"""
	def __init__(
		self,
		*,
		input_dim: int,
		output_dim: int,
		hidden_dim: list[int] = [8],
		activation: list[nn.Module] = [nn.ReLU()],
		loss_function: nn.Module | None = None,
		device: str = "cpu",
	) -> None:
		"""Initializes the FeedForward network with a sequential layout.

		Args:
			input_dim: Dimension of the input feature space.
			output_dim: Dimension of the output layer.
			hidden_dim: A list specifying the sizes of each hidden layer.
			activation: A list of activation modules; falls back to the last element if exhausted.
			loss_function: The loss function module to use. Defaults to SmoothL1Loss.
			device: The device to move the network to.
		"""
		layers = []
		curr_dim = input_dim
		activation_len = len(activation)

		for i, h_dim in enumerate(hidden_dim):
			layers.append(nn.Linear(curr_dim, h_dim))
			act = activation[i] if i < activation_len else activation[-1]
			layers.append(act)
			curr_dim = h_dim

		layers.append(nn.Linear(curr_dim, output_dim))
		sequential_nn = nn.Sequential(*layers)

		super().__init__(
			custom_nn=sequential_nn,
			loss_function=loss_function,
			device=device,
		)


	def copy(self) -> "FeedForward":
		"""Returns a deep copy of the FeedForward network instance."""
		# so i dont have to import copy in the other files so i can just use the static method 
		# _copy to copy the network
		return self._copy(self)
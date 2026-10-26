from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, Tuple, Optional

from shared.tools.config_base import OverWrightOperation, ConfigBase
from shared.utility import resolve_path


@dataclass
class CartPoleConfig(ConfigBase):
	episodes: int = 50_000
	hidden_dim: list[int] = field(default_factory=lambda: [8, 16])
	target_update_freq: int = 1000
	learning_rate: float = 0.001
	discount_factor: float = 0.99
	initial_epsilon: float = 1.0
	min_epsilon: float = 0.01
	epsilon_decay: float = 0.999675
	device: str = "cpu"
	satisfactory_reward_threshold: float = 475
	log_interval: int = 1000
	save_interval: int = 1000
	save_path: str = "models/cart_pole_dqn.pth"

	# Experience replay configuration
	replay_capacity: int = 100_000
	replay_batch_size: int = 64
	replay_warmup: int = 1000 

	CRITICAL_FIELDS: ClassVar[FrozenSet[str]] = frozenset({
		"hidden_dim",
		"device",
	})

	OVERWRITABLE_FIELDS: ClassVar[FrozenSet[str]] = frozenset({
		"log_interval",
		"save_interval",
		"save_path",
		"learning_rate",
		"target_update_freq",
		"discount_factor",
		"initial_epsilon",
		"min_epsilon",
		"epsilon_decay",
		"satisfactory_reward_threshold",
		"replay_capacity",
		"replay_batch_size",
	})

	SPECIAL_OVERWRITABLE_FIELDS: ClassVar[
		Dict[str, Tuple[OverWrightOperation, str, str]]
	] = {
		"episodes": (
			OverWrightOperation.GREATERTHAN,
			"episodes",
			"config__episodes",
		)
	}

	# Numerical bounds: (minimum, maximum)
	BOUNDS: ClassVar[
		Dict[str, Tuple[Optional[float], Optional[float]]]
	] = {
		"episodes": (1, None),
		"target_update_freq": (1, None),
		"learning_rate": (1e-6, 1.0),
		"discount_factor": (0.0, 1.0),
		"initial_epsilon": (0.0, 1.0),
		"min_epsilon": (0.0, 1.0),
		"epsilon_decay": (0.0, 1.0),
		"satisfactory_reward_threshold": (None, 500), # since max reward is 500 in CartPole-v1
		"log_interval": (1, None),
		"save_interval": (1, None),
		"replay_capacity": (1, None),
		"replay_batch_size": (0, None),
		"replay_warmup": (0, None),
	}

	def __post_init__(self) -> None:
		"""Validate network architecture and replay configuration."""
		if (
			not isinstance(self.hidden_dim, list)
			or not self.hidden_dim
			or not all(
				isinstance(d, int) and not isinstance(d, bool)
				and 1 <= d <= 4096
				for d in self.hidden_dim
			)
		):
			raise ValueError(
				"hidden_dim must be a non-empty list of integers "
				"between 1 and 4096."
			)

		if self.replay_batch_size > self.replay_capacity:
			raise ValueError(
				"replay_batch_size cannot exceed replay_capacity."
			)

		if self.min_epsilon > self.initial_epsilon:
			raise ValueError(
				"min_epsilon cannot exceed initial_epsilon."
			)

		if self.epsilon_decay <= 0.0 or self.epsilon_decay > 1.0:
			raise ValueError(
				"epsilon_decay must be greater than 0 and at most 1."
			)

	def resolved_save_path(self, current_dir: Path) -> Path:
		"""Resolve the model path relative to the supplied directory."""
		return resolve_path(current_dir, Path(self.save_path))
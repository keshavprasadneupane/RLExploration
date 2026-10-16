from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, Tuple, Optional
from shared.tools.config_base import OverWrightOperation, ConfigBase
from shared.validate import resolve_path


@dataclass
class MountainCarConfig(ConfigBase):
	episodes: int = 50_000
	hidden_dim: int = 64 
	target_update_freq: int = 500 
	learning_rate: float = 0.0005
	discount_factor: float = 0.99 
	initial_epsilon: float = 1.0
	min_epsilon: float = 0.01
	epsilon_decay: float = 0.995
	device: str = "cpu"
	satisfactory_reward_threshold: float = -110.0
	log_interval: int = 1000
	save_interval: int = 500
	save_path: str = "models/mountain_car_dqn.pth"

	CRITICAL_FIELDS: ClassVar[FrozenSet[str]] = frozenset({"hidden_dim", "device"})
	OVERWRITABLE_FIELDS: ClassVar[FrozenSet[str]] = frozenset({
		"log_interval",
		"learning_rate", 
		"target_update_freq", 
		"discount_factor", 
		"min_epsilon", 
		"epsilon_decay",
		"satisfactory_reward_threshold",
	})
	SPECIAL_OVERWRITABLE_FIELDS: ClassVar[Dict[str, Tuple[OverWrightOperation, str, str]]] = {
		"episodes": (
			OverWrightOperation.GREATERTHAN,
			"episodes",
			"config__episodes",
		)
	}

	# Numerical bound definitions (min_val, max_val, inclusive/exclusive or custom check)
	BOUNDS: ClassVar[Dict[str, Tuple[Optional[float], Optional[float]]]] = {
		"episodes": (1, None),
		"hidden_dim": (1, 4096),
		"target_update_freq": (1, None),
		"learning_rate": (1e-6, 1.0),
		"discount_factor": (0.0, 1.0),
		"initial_epsilon": (0.0, 1.0),
		"min_epsilon": (0.0, 1.0),
		"epsilon_decay": (0.0, 1.0),
		"satisfactory_reward_threshold": (None, 0.0),
		"log_interval": (1, None),
		"save_interval": (1, None),
	}

	def resolved_save_path(self, current_dir: Path) -> Path:
		"""Resolve the model path relative to the supplied directory."""
		return resolve_path(current_dir, Path(self.save_path))
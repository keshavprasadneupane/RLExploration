"""
Reusable base for training configs that can be validated and safely resumed.

ConfigBase gives any config dataclass two things:

1. Validation at creation time (BOUNDS and rule-overlap checks), so typos in a
   JSON file fail immediately instead of hours into a run.
2. Safe resuming from a checkpoint: you declare which fields are locked to the
   checkpoint, which may be re-tuned freely, and which may change only if
   training progress allows it.

Quick start
-----------
1. Subclass ConfigBase with @dataclass and declare your fields with defaults:

	@dataclass
	class MyConfig(ConfigBase):
		episodes: int = 50_000
		hidden_dim: int = 64
		learning_rate: float = 0.0005
		log_interval: int = 1_000

		CRITICAL_FIELDS = frozenset({"hidden_dim"})
		OVERWRITABLE_FIELDS = frozenset({"learning_rate", "log_interval"})
		SPECIAL_OVERWRITABLE_FIELDS = {
			"episodes": (OverWrightOperation.GREATERTHAN, "episodes", "config__episodes"),
		}
		BOUNDS = {"episodes": (1, None), "learning_rate": (1e-6, 1.0)}

   Annotate the rule sets as ClassVar[...] (see MountainCarConfig) so the
   dataclass decorator does not treat them as fields.

2. Save the config next to your training progress when you write a checkpoint:

	save_state["meta"] = {
		"episodes": completed_episodes,     # progress, used by SPECIAL rules
		"config": asdict(config),           # the effective config, required
	}

3. On start-up, load the JSON config, then resume like this:

	config = MyConfig(**json_data)
	meta = checkpoint["meta"]
	ok, mismatched = config.can_resume_with(meta["config"])
	if ok:
		config = config.merged_with_checkpoint(meta)   # effective config
	else:
		...  # critical mismatch: start a fresh model (or raise)

   Always save the *merged* config in later checkpoints. That is what makes a
   rejected value stay rejected on the next resume.

What each rule set does on resume
---------------------------------
CRITICAL_FIELDS
	Structural settings (network size, device, reward scale, ...). If the JSON
	value differs from the checkpoint's, can_resume_with returns
	(False, [names]) and you should not resume. A field missing from the
	checkpoint is skipped, so newly added critical fields do not block old
	checkpoints.

OVERWRITABLE_FIELDS
	Safe to retune between runs. The current (JSON) value always wins.

SPECIAL_OVERWRITABLE_FIELDS
	Overwritable only when a comparison against training progress allows it.
	Format: {field: (operation, progress_field, saved_field)}

	- progress_field: '__' path into the checkpoint meta data holding how far
	  training has gone ("episodes" = completed episodes).
	- saved_field: '__' path holding the value the checkpoint was saved with
	  ("config__episodes" = meta["config"]["episodes"]).
	- Rule: the new value is accepted only if operation(new_value, progress)
	  is True. Otherwise it is rejected, a message is printed, and the saved
	  value is kept.

	Reading the operation: it describes what the new value must be, compared
	with training progress, to be accepted ("accept if new <op> progress").

		GREATERTHAN         accept if new >  progress
		GREATERTHANOREQUAL  accept if new >= progress
		LESSTHAN            accept if new <  progress
		LESSTHANOREQUAL     accept if new <= progress

	Example with (GREATERTHAN, "episodes", "config__episodes"), 25,000 done:
		new episodes = 75,000 -> accepted (75,000 > 25,000)
		new episodes = 15,000 -> rejected, saved value (e.g. 50,000) is kept
		new episodes = 25,000 -> rejected (not greater, nothing new to run);
		                         use GREATERTHANOREQUAL to accept this case

	If progress_field, saved_field or the new value is missing/None, the rule
	cannot be evaluated and the new value is used.

BOUNDS
	{field: (min, max)}, both inclusive, None means unbounded on that side.
	Checked in __post_init__, so it runs for the JSON config and again for the
	merged config. Raises ValueError with the field name and limit.

Fields in none of the sets
	Restored from the checkpoint (the checkpoint's value wins over the JSON).
	Fields that the checkpoint does not contain (newly added) take the current
	value, and unknown keys in the checkpoint are ignored.

Notes
-----
- Overlaps (a field in two sets) emit a UserWarning. Precedence is
  CRITICAL > SPECIAL_OVERWRITABLE > OVERWRITABLE.
- If a subclass defines __post_init__, call super().__post_init__() first so
  bounds and rule checks still run.
- Add your own cross-field checks (e.g. min_epsilon <= initial_epsilon) in the
  subclass __post_init__; BOUNDS only checks one field at a time.
- Rule paths use '__' to reach into nested dicts, like sklearn parameters.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, FrozenSet, Self, Tuple, List
import warnings


class OverWrightOperation(Enum):
	LESSTHAN = 1
	LESSTHANOREQUAL = 2
	GREATERTHAN = 3
	GREATERTHANOREQUAL = 4


def operation_mapper(op: OverWrightOperation) -> Callable[[Any, Any], bool]:
	"""Map OverWrightOperation to a comparison function."""
	if op == OverWrightOperation.LESSTHAN:
		return lambda x, y: x < y
	elif op == OverWrightOperation.LESSTHANOREQUAL:
		return lambda x, y: x <= y
	elif op == OverWrightOperation.GREATERTHAN:
		return lambda x, y: x > y
	elif op == OverWrightOperation.GREATERTHANOREQUAL:
		return lambda x, y: x >= y
	raise ValueError(f"Unsupported operation: {op}")


def _fmt(value: Any) -> str:
	"""Format integers with thousands separators for readable messages."""
	if isinstance(value, int) and not isinstance(value, bool):
		return f"{value:,}"
	return str(value)


def _get_by_path(data: Any, path: str) -> Any:
	"""Retrieve a nested value from a dictionary using '__' as a separator (sklearn style)."""
	current: Any = data
	for part in path.split("__"):
		if isinstance(current, dict) and part in current:
			current = current[part]
		else:
			return None
	return current


def _set_by_path(data: Dict[str, Any], path: str, value: Any) -> None:
	"""Set a nested value in a dictionary using '__' as a separator."""
	parts = path.split("__")
	current = data
	for part in parts[:-1]:
		if part not in current or not isinstance(current[part], dict):
			current[part] = {}
		current = current[part]
	current[parts[-1]] = value


@dataclass
class ConfigBase:
	"""
	Base class providing checkpoint validation and merging logic for any configuration dataclass.

	Subclass it, declare fields, and fill in the class-level rule sets below.
	See the module docstring for a full walkthrough and examples.

	Public API:
		can_resume_with(saved_config) -> (ok, mismatched_fields)
		merged_with_checkpoint(meta_data) -> new config instance
	"""

	# Must match the checkpoint, otherwise training cannot resume.
	CRITICAL_FIELDS: ClassVar[FrozenSet[str]] = frozenset()

	# Always taken from the current config on resume.
	OVERWRITABLE_FIELDS: ClassVar[FrozenSet[str]] = frozenset()


	SPECIAL_OVERWRITABLE_FIELDS: ClassVar[Dict[str, Tuple[OverWrightOperation, str, str]]] = {}
	"""
		field_name: (current_field,(operation, progress_field, saved_field))
		
		Key: current config field name that is conditionally overwritable.
		operation: OverWrightOperation enum value describing the comparison.
		progress_field: '__' path into the checkpoint meta data holding how far
						training has progressed (e.g. "episodes" = completed episodes).
		saved_field:    '__' path into the checkpoint meta data holding the value
					the checkpoint was saved with (e.g. "config__episodes").

		Rule: the new value from the current config is accepted only if
		operation(new_value, progress_value) is True ("accept if new <op> progress").
		Otherwise the saved value is kept.
		#
		Example: (GREATERTHAN, "episodes", "config__episodes")
			completed=25_000, new=75_000 -> 75_000 > 25_000 is True  -> use 75_000
			completed=25_000, new=15_000 -> 15_000 > 25_000 is False -> keep saved value
	"""
	BOUNDS: ClassVar[Dict[str, Tuple[Any, Any]]] = {}

	def __post_init__(self) -> None:
		"""Validate rule configurations and bounds upon instantiation."""
		self._validate_rules()
		self._validate_bounds()

	def _validate_bounds(self) -> None:
		"""Validate that numerical config fields fall within acceptable bounds."""
		for field_name, (min_val, max_val) in self.BOUNDS.items():
			if hasattr(self, field_name):
				val = getattr(self, field_name)
				if val is not None:
					if min_val is not None and val < min_val:
						raise ValueError(f"Config field '{field_name}' value {val} is below minimum bound {min_val}.")
					if max_val is not None and val > max_val:
						raise ValueError(f"Config field '{field_name}' value {val} exceeds maximum bound {max_val}.")

	def _validate_rules(self) -> None:
		"""Warn about overlapping fields and prioritize safety/precedence."""
		overlapping_fields: FrozenSet[str] = self.OVERWRITABLE_FIELDS.intersection(
			self.SPECIAL_OVERWRITABLE_FIELDS.keys()
		)
		if overlapping_fields:
			warnings.warn(
				f"Fields {overlapping_fields} are present in both "
				"OVERWRITABLE_FIELDS and SPECIAL_OVERWRITABLE_FIELDS. "
				"SPECIAL_OVERWRITABLE_FIELDS will take precedence.",
				UserWarning,
				stacklevel=2,
			)

		critical_overlaps: FrozenSet[str] = self.CRITICAL_FIELDS.intersection(
			self.OVERWRITABLE_FIELDS.union(self.SPECIAL_OVERWRITABLE_FIELDS.keys())
		)
		if critical_overlaps:
			warnings.warn(
				f"Fields {critical_overlaps} are present in both CRITICAL_FIELDS and "
				"OVERWRITABLE_FIELDS/SPECIAL_OVERWRITABLE_FIELDS. "
				"Critical fields will take precedence.",
				UserWarning,
				stacklevel=2,
			)

	def can_resume_with(self, saved_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
		"""
		Check whether the checkpoint is compatible with this config.

		Args:
			saved_config: the config dict stored in the checkpoint (meta["config"]).

		Returns:
			(ok, mismatched_fields): ok is False if any CRITICAL field present in
			the checkpoint differs from this config. mismatched_fields is sorted.
		"""
		mismatched_fields: List[str] = [
			name for name in self.CRITICAL_FIELDS
			if name in saved_config and saved_config[name] != getattr(self, name)
		]
		return not mismatched_fields, sorted(mismatched_fields)

	def merged_with_checkpoint(self, meta_data: Dict[str, Any]) -> Self:
		"""
		Build the effective config for resuming from a checkpoint.

		Per field:
		  - CRITICAL: current value (already verified equal by can_resume_with).
		  - OVERWRITABLE: current value always wins.
		  - SPECIAL_OVERWRITABLE: current value wins only if it satisfies the rule
			against training progress (new <op> progress), otherwise the
			checkpoint's saved value is kept.
		  - everything else: restored from the checkpoint.
		  - fields the checkpoint doesn't know about (newly added): current value.

		Args:
			meta_data: the checkpoint meta dict. It must hold the saved config under
				"config" and any progress values referenced by
				SPECIAL_OVERWRITABLE_FIELDS (e.g. "episodes").

		Returns a new instance of the exact child config class. Call
		can_resume_with first; this method does not re-check CRITICAL fields.
		"""
		saved_config: Any = meta_data.get("config", {})
		if not isinstance(saved_config, dict):
			raise ValueError("Checkpoint configuration must be a dictionary.")

		current_values: Dict[str, Any] = asdict(self)
		locked_to_current: FrozenSet[str] = (
			self.OVERWRITABLE_FIELDS
			| frozenset(self.SPECIAL_OVERWRITABLE_FIELDS.keys())
			| self.CRITICAL_FIELDS
		)

		# Start from the current config, then restore plain fields from the checkpoint.
		# Unknown keys in the checkpoint are ignored instead of crashing the constructor.
		merged: Dict[str, Any] = dict(current_values)
		for name, saved_value in saved_config.items():
			if name in current_values and name not in locked_to_current:
				merged[name] = saved_value

		# Conditional overrides: accept the new value only if it satisfies the
		# rule against training progress (operation(new_value, progress_value)).
		for name, (operation, progress_field, saved_field) in self.SPECIAL_OVERWRITABLE_FIELDS.items():
			if name in self.CRITICAL_FIELDS or name not in current_values:
				continue

			new_value: Any = current_values[name]
			progress_value: Any = _get_by_path(meta_data, progress_field)
			saved_value = _get_by_path(meta_data, saved_field)

			# Without both reference values the rule can't be evaluated; keep the new value.
			if new_value is None or progress_value is None or saved_value is None:
				continue

			if not operation_mapper(operation)(new_value, progress_value):
				print(
					f"  Ignoring new '{name}'={_fmt(new_value)}: checkpoint progress is "
					f"{_fmt(progress_value)}. Keeping saved value {_fmt(saved_value)}."
				)
				merged[name] = saved_value

		return type(self)(**merged)
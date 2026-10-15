
import shared.tools.execution_timer  # noqa: F401 — automatic timer

import argparse
import dataclasses
import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

from shared.agent.simple_dqn_agent import SimpleDQNTrainAgent
from shared.mapper.dicrete_mapper import DiscreteActionMapper
from shared.network.feed_forward import FeedForward
from shared.wrapper import stop_by_user


CURRENT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
	"""Configuration dataclass mirroring the JSON config structure."""

	episodes: int = 50_000_000
	hidden_dim: int = 32
	output_dim: int = 21
	target_update_freq: int = 1000
	learning_rate: float = 0.01
	discount_factor: float = 0.99
	initial_epsilon: float = 1.0
	min_epsilon: float = 0.1
	epsilon_decay: float = 0.9999
	device: str = "cpu"
	early_exit_reward: float = -100.0
	early_exit_count: int = 100
	log_interval: int = 1000
	save_interval: int = 1_000
	save_path: str = "models/pendulum_dqn.pth"

	CRITICAL_FIELDS: ClassVar[frozenset[str]] = frozenset(
		{"hidden_dim", "output_dim", "device"}
	)
	OVERWRITABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
		{"early_exit_count", "log_interval", "save_interval"}
	)

	@property
	def resolved_save_path(self) -> Path:
		"""Resolve the model path relative to this file."""
		path = Path(self.save_path)
		return path if path.is_absolute() else CURRENT_DIR / path

	def can_resume_with(
		self,
		saved_config: dict[str, Any],
	) -> tuple[bool, list[str]]:
		"""Check whether the saved model architecture is compatible."""
		mismatched_fields = [
			name
			for name in self.CRITICAL_FIELDS
			if name in saved_config
			and saved_config[name] != getattr(self, name)
		]

		return not mismatched_fields, sorted(mismatched_fields)

	def merged_with_checkpoint(
		self,
		saved_config: dict[str, Any],
	) -> "Config":
		"""
		Restore checkpoint settings while retaining current overwritable fields.

		Critical fields must be checked with can_resume_with() first.
		"""
		current_values = asdict(self)

		for name, value in saved_config.items():
			if (
				name in current_values
				and name not in self.OVERWRITABLE_FIELDS
			):
				current_values[name] = value

		return Config(**current_values)


@dataclass
class TrainingState:
	"""Mutable state shared by training and interruption handlers."""

	epsilon: float
	completed_episodes: int = 0
	total_reward_sum: float = 0.0
	successful_episodes: int = 0
	loss_sum: float = 0.0
	loss_count: int = 0
	recent_rewards: deque[float] = field(
		default_factory=lambda: deque(maxlen=1000)
	)


def load_config(config_path: Path) -> Config | None:
	"""Load a JSON configuration file, returning None if it is missing."""
	resolved_path = (
		CURRENT_DIR / config_path
		if not config_path.is_absolute()
		else config_path
	)

	if not resolved_path.exists():
		return None

	with resolved_path.open("r", encoding="utf-8") as file:
		data = json.load(file)

	if not isinstance(data, dict):
		raise ValueError(
			f"Configuration must contain a JSON object: {resolved_path}"
		)

	valid_keys = {item.name for item in dataclasses.fields(Config)}
	filtered_data = {
		key: value
		for key, value in data.items()
		if key in valid_keys
	}

	return Config(**filtered_data)


def create_default_config(config_path: Path) -> Path:
	"""Create a default JSON configuration file."""
	resolved_path = (
		CURRENT_DIR / config_path
		if not config_path.is_absolute()
		else config_path
	)

	config = Config()
	resolved_path.parent.mkdir(parents=True, exist_ok=True)

	with resolved_path.open("w", encoding="utf-8") as file:
		json.dump(asdict(config), file, indent=4)

	return resolved_path


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description=(
			"Train a DQN agent on Pendulum-v1 "
			"using a JSON configuration file."
		)
	)

	parser.add_argument(
		"--config",
		type=Path,
		default=Path("config.json"),
		help="Path to the JSON configuration file.",
	)
	parser.add_argument(
		"--new-model",
		action="store_true",
		help="Create a new model instead of resuming a checkpoint.",
	)

	return parser.parse_args()


def create_agent(
	env: gym.Env,
	config: Config,
) -> tuple[SimpleDQNTrainAgent, DiscreteActionMapper]:
	"""Create the DQN agent and its discrete action mapper."""
	if not isinstance(env.observation_space, gym.spaces.Box):
		raise ValueError("Expected observation space to be Box.")

	if not isinstance(env.action_space, gym.spaces.Box):
		raise ValueError("Expected action space to be Box.")

	if len(env.observation_space.shape) != 1:
		raise ValueError(
			"Expected a one-dimensional observation space."
		)

	action_mapper = DiscreteActionMapper(
		env.action_space,
		num_actions=config.output_dim,
	)

	network = FeedForward(
		input_dim=env.observation_space.shape[0],
		hidden_dim=[config.hidden_dim],
		output_dim=config.output_dim,
		device=config.device,
	)

	optimizer = torch.optim.Adam(
		network.parameters(),
		lr=config.learning_rate,
	)

	agent = SimpleDQNTrainAgent(
		network=network,
		optimizer=optimizer,
		target_update_freq=config.target_update_freq,
	)

	return agent, action_mapper


def save_model(
	config: Config,
	agent: SimpleDQNTrainAgent,
	state: TrainingState,
) -> None:
	"""Save the model, optimizer state, training progress, and config."""
	save_path = config.resolved_save_path
	save_path.parent.mkdir(parents=True, exist_ok=True)

	save_state = agent.get_save_state()
	save_state.update(
		{
			"episodes": state.completed_episodes,
			"epsilon": state.epsilon,
			"steps": agent.steps,
			"total_reward_sum": state.total_reward_sum,
			"successful_episodes": state.successful_episodes,
			"config": asdict(config),
		}
	)

	torch.save(save_state, save_path)


def train_loop(
	config: Config,
	env: gym.Env,
	agent: SimpleDQNTrainAgent,
	action_mapper: DiscreteActionMapper,
	state: TrainingState,
) -> TrainingState:
	"""Train the agent while updating the shared training state."""
	for episode in range(
		state.completed_episodes + 1,
		config.episodes + 1,
	):
		observation, _ = env.reset()
		terminated = False
		truncated = False
		total_reward = 0.0

		while not (terminated or truncated):
			if random.random() < state.epsilon:
				action, action_value = (
					action_mapper.sample_random_action()
				)
			else:
				action, _ = agent.get_action(observation)
				action_value = action_mapper.to_continuous(action)

			(
				next_observation,
				reward,
				terminated,
				truncated,
				_,
			) = env.step(action_value)

			reward = float(reward)

			loss = agent.update_weights(
				state=observation,
				action=action,
				reward=reward,
				next_state=next_observation,
				done=terminated,
				discount_factor=config.discount_factor,
			)

			state.loss_sum += loss
			state.loss_count += 1
			total_reward += reward
			observation = next_observation

		state.completed_episodes = episode
		state.total_reward_sum += total_reward
		state.recent_rewards.append(total_reward)

		successful = total_reward >= config.early_exit_reward
		if successful:
			state.successful_episodes += 1

		state.epsilon = max(
			config.min_epsilon,
			state.epsilon * config.epsilon_decay,
		)

		if episode % config.log_interval == 0:
			rewards = np.asarray(
				state.recent_rewards,
				dtype=np.float64,
			)

			average_loss = (
				state.loss_sum / state.loss_count
				if state.loss_count
				else 0.0
			)

			if rewards.size:
				print(
					f"Episode {episode:,} | "
					f"Epsilon: {state.epsilon:.5f} | "
					f"Recent average reward: {rewards.mean():.2f} | "
					f"Recent std: {rewards.std():.2f} | "
					f"Recent min: {rewards.min():.2f} | "
					f"Recent max: {rewards.max():.2f} | "
					f"Average loss: {average_loss:.6f}"
				)

			state.loss_sum = 0.0
			state.loss_count = 0

		if successful:
			print(
				f"Episode {episode:,}: "
				f"reward={total_reward:.1f}, "
				f"successful episodes={state.successful_episodes}"
			)

		if episode % config.save_interval == 0:
			save_model(config, agent, state)
			print(
				f"Checkpoint saved at episode {episode:,}: "
				f"{config.resolved_save_path}"
			)

		if state.successful_episodes >= config.early_exit_count:
			print(
				f"Early exit at episode {episode:,}: "
				f"{state.successful_episodes} episodes reached "
				f"the reward threshold of {config.early_exit_reward}."
			)
			break

	return state


def print_config(config: Config, state: TrainingState) -> None:
	"""Print the effective training configuration."""
	print(
		f"Training for up to {config.episodes:,} episodes\n"
		f"Hidden dimension: {config.hidden_dim}\n"
		f"Output dimension: {config.output_dim}\n"
		f"Target update frequency: {config.target_update_freq}\n"
		f"Learning rate: {config.learning_rate}\n"
		f"Discount factor: {config.discount_factor}\n"
		f"Initial/current epsilon: {state.epsilon}\n"
		f"Minimum epsilon: {config.min_epsilon}\n"
		f"Epsilon decay: {config.epsilon_decay}\n"
		f"Device: {config.device}\n"
		f"Early-exit reward: {config.early_exit_reward}\n"
		f"Early-exit count: {config.early_exit_count}\n"
		f"Log interval: {config.log_interval}\n"
		f"Save interval: {config.save_interval}\n"
		f"Save path: {config.resolved_save_path}\n"
	)


def main() -> None:
	"""Load configuration, resume if possible, and train the agent."""
	args = parse_args()
	config = load_config(args.config)

	if config is None:
		created_path = create_default_config(args.config)
		print(
			"Configuration file was not found. "
			f"Default configuration created at: {created_path}"
		)
		return

	env = gym.make("Pendulum-v1")

	try:
		state = TrainingState(epsilon=config.initial_epsilon)
		checkpoint: dict[str, Any] | None = None
		save_path = config.resolved_save_path

		if not args.new_model and save_path.exists():
			print(f"Checking existing checkpoint at {save_path}...")

			loaded = torch.load(
				save_path,
				map_location=config.device,
				weights_only=False,
			)

			if not isinstance(loaded, dict):
				raise ValueError("Invalid checkpoint format.")

			saved_config = loaded.get("config", {})
			if not isinstance(saved_config, dict):
				raise ValueError(
					"Checkpoint configuration must be a dictionary."
				)

			can_resume, mismatched_fields = config.can_resume_with(
				saved_config
			)

			if can_resume:
				# Retain the current JSON value for early_exit_count.
				# Restore every other recognized setting from the checkpoint.
				current_early_exit_count = config.early_exit_count

				config = config.merged_with_checkpoint(saved_config)
				config.early_exit_count = current_early_exit_count

				checkpoint = loaded

				print("Checkpoint architecture is compatible.")
				print("Restoring saved training configuration.")
				print(
					"Using current early_exit_count from JSON: "
					f"{config.early_exit_count}"
				)
			else:
				print(
					"Critical configuration mismatch detected: "
					f"{mismatched_fields}"
				)
				print("Starting a fresh model.")

		elif args.new_model:
			print("--new-model specified. Creating a fresh model.")
		else:
			print("No existing model found. Creating a fresh model.")

		# Build the agent using the effective configuration.
		agent, action_mapper = create_agent(
			env=env,
			config=config,
		)

		if checkpoint is not None:
			agent.load_save_state(checkpoint)

			state.completed_episodes = checkpoint.get("episodes", 0)
			state.epsilon = checkpoint.get(
				"epsilon",
				config.initial_epsilon,
			)
			state.total_reward_sum = checkpoint.get(
				"total_reward_sum",
				0.0,
			)
			state.successful_episodes = checkpoint.get(
				"successful_episodes",
				0,
			)

			print(
				"Resuming training from episode "
				f"{state.completed_episodes:,}."
			)

		interrupted_by_user = False

		def interrupt_handler() -> None:
			nonlocal interrupted_by_user
			interrupted_by_user = True

			#save_model(config, agent, state)

			print(
				"\nTraining interrupted by user at episode "
				f"{state.completed_episodes:,}."
			)
			# print(
			# 	f"Model checkpoint saved to: "
			# 	f"{config.resolved_save_path}"
			# )

		print_config(config, state)

		stop_by_user(
			train_loop,
			post_exception=interrupt_handler,
			message="Training interrupted by user.",
			config=config,
			env=env,
			agent=agent,
			action_mapper=action_mapper,
			state=state,
		)

		if not interrupted_by_user:
			save_model(config, agent, state)

			average_reward = (
				state.total_reward_sum / state.completed_episodes
				if state.completed_episodes
				else 0.0
			)

			print(
				f"\nTraining completed: "
				f"{state.completed_episodes:,} episodes"
			)
			print(f"Overall average reward: {average_reward:.2f}")
			print(
				f"Successful episodes: "
				f"{state.successful_episodes}"
			)
			print(f"Final epsilon: {state.epsilon:.5f}")
			print(
				f"Model checkpoint saved to: "
				f"{config.resolved_save_path}"
			)

	finally:
		env.close()


if __name__ == "__main__":
	main()

import shared.tools.execution_timer  # noqa: F401 — automatic timer

import argparse
import dataclasses
import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
	"""Configuration dataclass mirroring JSON config structure."""

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
	save_interval: int = 10_000
	save_path: str = "models/pendulum_dqn.pth"

	@property
	def resolved_save_path(self) -> Path:
		return CURRENT_DIR / self.save_path if not Path(self.save_path).is_absolute() else Path(self.save_path)


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
	"""Load configuration from JSON file. Returns None if file does not exist."""
	resolved_path = CURRENT_DIR / config_path if not config_path.is_absolute() else config_path
	
	if resolved_path.exists():
		with open(resolved_path, "r") as f:
			data = json.load(f)
			valid_keys = {f.name for f in dataclasses.fields(Config)}
			filtered_data = {k: v for k, v in data.items() if k in valid_keys}
			return Config(**filtered_data)
	return None


def create_default_config(config_path: Path) -> Path:
	"""Create a default configuration JSON file."""
	resolved_path = CURRENT_DIR / config_path if not config_path.is_absolute() else config_path
	config = Config()
	resolved_path.parent.mkdir(parents=True, exist_ok=True)
	with open(resolved_path, "w") as f:
		json.dump(asdict(config), f, indent=4)
	return resolved_path


def parse_args() -> argparse.Namespace:
	"""Parse terminal arguments for config path and new model flag."""
	parser = argparse.ArgumentParser(
		description="Train a DQN agent on Pendulum-v1 using a JSON config file."
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
		default=False,
		help="Force creation of a new model instead of attempting to resume.",
	)

	return parser.parse_args()


def create_agent(
	env: gym.Env,
	config: Config,
) -> tuple[SimpleDQNTrainAgent, DiscreteActionMapper]:
	"""Create the DQN agent and discrete action mapper."""

	if not isinstance(env.observation_space, gym.spaces.Box):
		raise ValueError("Expected observation space to be Box.")

	if not isinstance(env.action_space, gym.spaces.Box):
		raise ValueError("Expected action space to be Box.")

	if len(env.observation_space.shape) != 1:
		raise ValueError("Expected a one-dimensional observation space.")

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
	"""Save the model and current training metadata including config dictionary."""

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
	"""Train the agent while updating shared training state."""

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
	"""Print the training configuration."""

	print(
		f"Training for up to {config.episodes:,} episodes\n"
		f"Hidden dimension: {config.hidden_dim}\n"
		f"Output dimension: {config.output_dim}\n"
		f"Target update frequency: {config.target_update_freq}\n"
		f"Learning rate: {config.learning_rate}\n"
		f"Discount factor: {config.discount_factor}\n"
		f"Initial epsilon: {state.epsilon}\n"
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
	args = parse_args()
	config = load_config(args.config)

	if config is None:
		created_path = create_default_config(args.config)
		print(f"Configuration file was not found. Default configuration created at: {created_path}")
		return

	env = gym.make("Pendulum-v1")

	try:
		agent, action_mapper = create_agent(env=env, config=config)
		state = TrainingState(epsilon=config.initial_epsilon)

		save_path = config.resolved_save_path

		if not args.new_model and save_path.exists():
			print(f"Checking existing checkpoint at {save_path}...")
			checkpoint = torch.load(save_path, map_location=config.device)
			saved_config = checkpoint.get("config", {})

			current_config_dict = asdict(config)
			if "save_path" in saved_config:
				saved_config["save_path"] = str(saved_config["save_path"])
			current_config_dict["save_path"] = str(current_config_dict["save_path"])

			if saved_config == current_config_dict:
				agent.load_save_state(checkpoint)
				state.completed_episodes = checkpoint.get("episodes", 0)
				state.epsilon = checkpoint.get("epsilon", config.initial_epsilon)
				state.total_reward_sum = checkpoint.get("total_reward_sum", 0.0)
				state.successful_episodes = checkpoint.get("successful_episodes", 0)
				print(f"Configuration matches! Resuming training from episode {state.completed_episodes:,}.")
			else:
				print("Configuration mismatch detected between checkpoint and configuration file!")
				print("Starting fresh model creation due to config mismatch.")
		else:
			if args.new_model:
				print("--new-model flag specified. Creating a fresh model.")
			else:
				print("No existing model found. Creating a fresh model.")

		interrupted_by_user = False

		def interrupt_handler() -> None:
			nonlocal interrupted_by_user
			interrupted_by_user = True

			save_model(config, agent, state)

			print(
				"\nTraining interrupted by user at episode "
				f"{state.completed_episodes:,}."
			)
			print(f"Model checkpoint saved to: {config.resolved_save_path}")

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
			print(f"Model checkpoint saved to: {config.resolved_save_path}")

	finally:
		env.close()


if __name__ == "__main__":
	main()
import shared.tools.execution_timer  # noqa: F401 — automatic timer

import argparse
import dataclasses
import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from shared.agent.simple_dqn_agent import SimpleDQNTrainAgent
from shared.network.feed_forward import FeedForward
from shared.wrapper import stop_by_user
from .config import MountainCarConfig as Config

CURRENT_DIR = Path(__file__).resolve().parent


@dataclass
class TrainingState:
	"""Mutable state shared by training and interruption handlers."""

	epsilon: float
	completed_episodes: int = 0
	total_reward_sum: float = 0.0
	successful_episodes_window: int = 0
	total_successful_episodes: int = 0
	loss_sum: float = 0.0
	loss_count: int = 0
	recent_rewards: deque[float] = field(default_factory=deque)


def resolve_path(path: Path) -> Path:
	"""Resolve relative paths against the directory containing this file."""
	return path if path.is_absolute() else CURRENT_DIR / path


def load_config(config_path: Path) -> Config | None:
	"""Load a JSON configuration file, returning None if it is missing."""
	resolved_path = resolve_path(config_path)

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

def episode_to_reach_min_epsilon(y: float,x:float,a: float) -> int:
	"""Calculate the number of episodes to reach the minimum epsilon."""
	return int(np.log(y / x) / np.log(a))

def create_default_config(config_path: Path) -> Path:
	"""Create a default JSON configuration file."""
	resolved_path = resolve_path(config_path)
	resolved_path.parent.mkdir(parents=True, exist_ok=True)

	with resolved_path.open("w", encoding="utf-8") as file:
		json.dump(asdict(Config()), file, indent=4)

	return resolved_path


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description=(
			"Train a DQN agent on MountainCar-v0 "
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
) -> SimpleDQNTrainAgent:
	"""Create the DQN agent."""
	if not isinstance(env.observation_space, gym.spaces.Box):
		raise ValueError("Expected observation space to be Box.")

	if not isinstance(env.action_space, gym.spaces.Discrete):
		raise ValueError("Expected action space to be Discrete.")

	if len(env.observation_space.shape) != 1:
		raise ValueError(
			"Expected a one-dimensional observation space."
		)

	network = FeedForward(
		input_dim=env.observation_space.shape[0],
		hidden_dim=[config.hidden_dim],
		output_dim=env.action_space.n,
		device=config.device,
	)

	optimizer = torch.optim.Adam(
		network.parameters(),
		lr=config.learning_rate,
	)

	return SimpleDQNTrainAgent(
		network=network,
		optimizer=optimizer,
		target_update_freq=config.target_update_freq,
	)


def save_model(
	config: Config,
	agent: SimpleDQNTrainAgent,
	state: TrainingState,
) -> None:
	"""Save the model, optimizer state, training progress, and config."""
	save_path = config.resolved_save_path(CURRENT_DIR)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	save_state = agent.get_save_state()
	save_state["meta"] = {
		"episodes": state.completed_episodes,
		"epsilon": state.epsilon,
		"steps": getattr(agent, "steps", 0),
		"total_reward_sum": state.total_reward_sum,
		"total_successful_episodes": state.total_successful_episodes,
		"config": asdict(config),
	}

	torch.save(save_state, save_path)

def calculate_mountain_car_reward(
    reward: float,
    observation: np.ndarray,
    next_observation: np.ndarray,
    terminated: bool,
    gamma: float = 0.99,
    position_weight: float = 1.0,
    velocity_weight: float = 0.1,
    terminal_bonus: float = 100.0,
) -> float:
    """Calculate potential-based shaped reward for MountainCar-v0."""

    min_position = -1.2
    goal_position = 0.5
    max_speed = 0.07

    def potential(state: np.ndarray) -> float:
        position = float(state[0])
        velocity = float(state[1])

        normalized_position = (
            (position - min_position)
            / (goal_position - min_position)
        )
        normalized_position = max(0.0, min(1.0, normalized_position))

        normalized_velocity = velocity / max_speed

        return (
            position_weight * normalized_position
            + velocity_weight * normalized_velocity**2
        )

    previous_potential = potential(observation)
    next_potential = potential(next_observation)

    shaped_reward = (
        float(reward)
        + gamma * next_potential
        - previous_potential
    )

    if terminated:
        shaped_reward += terminal_bonus

    return shaped_reward

def train_loop(
	config: Config,
	env: gym.Env,
	agent: SimpleDQNTrainAgent,
	state: TrainingState,
) -> TrainingState:
	"""Train the agent up to total episodes while logging window statistics."""
	# Ensure deque tracks the log interval window size dynamically
	if state.recent_rewards.maxlen != config.log_interval:
		state.recent_rewards = deque(state.recent_rewards, maxlen=config.log_interval)

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
				action = env.action_space.sample()
			else:
				action, _ = agent.get_action(observation)

			(
				next_observation,
				reward,
				terminated,
				truncated,
				_,
			) = env.step(action)

			reward = float(reward)
			reward = calculate_mountain_car_reward(
				reward=reward,
				observation=observation,
				next_observation=next_observation,
				terminated=terminated,
				gamma=config.discount_factor,
				position_weight=1.0,
				velocity_weight=0.1,
				terminal_bonus=100.0,
			)
	
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

		if total_reward >= config.satisfactory_reward_threshold:
			state.successful_episodes_window += 1
			state.total_successful_episodes += 1

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
					f"Window Satisfactory Reached: {state.successful_episodes_window}/{config.log_interval} | "
					f"Recent window avg reward: {rewards.mean():.2f} | "
					f"Window min: {rewards.min():.2f} | "
					f"Window max: {rewards.max():.2f} | "
					f"Average loss: {average_loss:.6f}"
				)

			state.loss_sum = 0.0
			state.loss_count = 0
			state.successful_episodes_window = 0  # Reset window counter

		if episode % config.save_interval == 0:
			save_model(config, agent, state)
			print(
				f"Checkpoint saved at episode {episode:,}: "
				f"{config.resolved_save_path(CURRENT_DIR)}"
			)

	return state


def print_config(config: Config, state: TrainingState) -> None:
	"""Print the effective training configuration."""
	print(
		"Run\n"
		f"  Max episodes: {config.episodes:,}\n"
		f"  Device: {config.device}\n"
		"\n"
		"Model & optimizer\n"
		f"  Hidden dimension: {config.hidden_dim}\n"
		f"  Learning rate: {config.learning_rate}\n"
		f"  Discount factor: {config.discount_factor}\n"
		f"  Target update frequency: {config.target_update_freq}\n"
		"\n"
		"Exploration\n"
		f"  Current epsilon: {state.epsilon}\n"
		f"  Minimum epsilon: {config.min_epsilon}\n"
		f"  Epsilon decay: {config.epsilon_decay}\n"
		f"  Runs to reach min epsilon: {episode_to_reach_min_epsilon(config.min_epsilon, config.initial_epsilon, config.epsilon_decay)}\n"
		"\n"
		"Logging & checkpoints\n"
		f"  Satisfactory reward threshold: {config.satisfactory_reward_threshold}\n"
		f"  Log interval: {config.log_interval}\n"
		f"  Save interval: {config.save_interval}\n"
		f"  Save path: {config.resolved_save_path(CURRENT_DIR)}"
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

	env = gym.make("MountainCar-v0")

	try:
		checkpoint: dict[str, Any] | None = None
		save_path = config.resolved_save_path(CURRENT_DIR)
		state = TrainingState(
			epsilon=config.initial_epsilon,
			recent_rewards=deque(maxlen=config.log_interval)
		)

		if not args.new_model and save_path.exists():
			print(f"Checking existing checkpoint: {save_path}")

			loaded = torch.load(
				save_path,
				map_location=config.device,
				weights_only=False,
			)

			if not isinstance(loaded, dict):
				raise ValueError("Invalid checkpoint format.")

			meta_data = loaded.get("meta", {})
			if not isinstance(meta_data, dict):
				raise ValueError("Checkpoint metadata must be a dictionary.")

			saved_config = meta_data.get("config", {})
			if not isinstance(saved_config, dict):
				raise ValueError(
					"Checkpoint configuration must be a dictionary."
				)

			can_resume, mismatched_fields = config.can_resume_with(saved_config)

			if can_resume:
				config = config.merged_with_checkpoint(meta_data)
				checkpoint = loaded

				print(
					"  Compatible. Overwritable fields come from the JSON "
					"config; the rest are restored from the checkpoint."
				)
			else:
				print(
					"  Critical configuration mismatch detected: "
					f"{mismatched_fields}"
				)
				print("  Starting a fresh model.")

		elif args.new_model:
			print("--new-model specified. Creating a fresh model.")
		else:
			print("No existing model found. Creating a fresh model.")

		agent = create_agent(env=env, config=config)

		if checkpoint is not None:
			agent.load_save_state(checkpoint)

			meta_data = checkpoint["meta"]

			# Cast to plain types: checkpoints written before the reward fix may
			# hold numpy scalars, which would otherwise be saved again.
			state.completed_episodes = int(meta_data.get("episodes", 0))
			state.epsilon = float(
				meta_data.get(
					"epsilon",
					config.initial_epsilon,
				)
			)
			state.total_reward_sum = float(
				meta_data.get(
					"total_reward_sum",
					0.0,
				)
			)
			state.total_successful_episodes = int(
				meta_data.get(
					"total_successful_episodes",
					0,
				)
			)

			print(
				"  Resuming training from episode "
				f"{state.completed_episodes:,}."
			)

		interrupted_by_user = False

		def interrupt_handler() -> None:
			nonlocal interrupted_by_user
			interrupted_by_user = True

			print(
				"\nTraining interrupted by user at episode "
				f"{state.completed_episodes:,}."
			)

		print()
		print_config(config, state)
		print()

		stop_by_user(
			train_loop,
			post_exception=interrupt_handler,
			message="Training interrupted by user.",
			config=config,
			env=env,
			agent=agent,
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
				f"Total successful episodes (>= {config.satisfactory_reward_threshold}): "
				f"{state.total_successful_episodes}"
			)
			print(f"Final epsilon: {state.epsilon:.5f}")
			print(
				"Model checkpoint saved to: "
				f"{config.resolved_save_path(CURRENT_DIR)}"
			)

	finally:
		env.close()


if __name__ == "__main__":
	main()

import shared.tools.execution_timer  # noqa: F401 — automatic timer

import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from shared.agent.dqn_agent import DQNetworkTrainAgent
from shared.network.feed_forward import FeedForward
from shared.wrapper import stop_by_user
from shared.utility import (
	episode_to_reach_min_epsilon,
	load_config,
	create_default_config,
	save_the_model,
	load_checkpoint,
	parse_args,
)
from .config import CartPoleConfig as Config


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


def create_agent(
	env: gym.Env,
	config: Config,
) -> DQNetworkTrainAgent:
	"""Create the DQN network, optimizer, and optional replay buffer."""

	if not isinstance(env.observation_space, gym.spaces.Box):
		raise ValueError("Expected observation space to be Box.")

	if not isinstance(env.action_space, gym.spaces.Discrete):
		raise ValueError("Expected action space to be Discrete.")

	if len(env.observation_space.shape) != 1:
		raise ValueError("Expected a one-dimensional observation space.")

	hidden_dims: list[int] = (
		config.hidden_dim
		if isinstance(config.hidden_dim, list)
		else [config.hidden_dim]
	)

	network = FeedForward(
		input_dim=env.observation_space.shape[0],
		hidden_dim=hidden_dims,
		output_dim=env.action_space.n,
		device=config.device,
	)

	optimizer = torch.optim.Adam(
		network.parameters(),
		lr=config.learning_rate,
	)

	return DQNetworkTrainAgent(
		network=network,
		optimizer=optimizer,
		target_update_freq=config.target_update_freq,
		replay_capacity=config.replay_capacity,
		batch_size=config.replay_batch_size,
		warmup=config.replay_warmup,
	)


def get_replay_size(agent: DQNetworkTrainAgent) -> int:
	"""Return the number of stored transitions, or zero without replay."""

	if agent.replay_buffer is None:
		return 0

	return len(agent.replay_buffer)


def get_replay_description(
	agent: DQNetworkTrainAgent,
	config: Config,
) -> str:
	"""Format replay statistics for logging."""

	if not agent.replay:
		return "Replay: disabled"

	return (
		f"Replay: {get_replay_size(agent):,}/"
		f"{config.replay_capacity:,}"
	)


def save_model(
	config: Config,
	agent: DQNetworkTrainAgent,
	state: TrainingState,
) -> None:
	"""Save model, optimizer, optional replay memory, and progress."""

	save_path = config.resolved_save_path(CURRENT_DIR)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	save_state = agent.get_save_state()

	save_state["meta"] = {
		"episodes": int(state.completed_episodes),
		"epsilon": float(state.epsilon),
		"steps": int(agent.steps),
		"total_reward_sum": float(state.total_reward_sum),
		"total_successful_episodes": int(
			state.total_successful_episodes
		),
		"config": asdict(config),
	}

	save_the_model(save_path, save_state)


def train_loop(
	config: Config,
	env: gym.Env,
	agent: DQNetworkTrainAgent,
	state: TrainingState,
) -> TrainingState:
	"""Train up to the configured maximum episode count."""

	if state.recent_rewards.maxlen != config.log_interval:
		state.recent_rewards = deque(
			state.recent_rewards,
			maxlen=config.log_interval,
		)

	for episode in range(
		state.completed_episodes + 1,
		config.episodes + 1,
	):
		state_obs, _ = env.reset()
		terminated = False
		truncated = False
		total_reward = 0.0

		while not (terminated or truncated):
			if random.random() < state.epsilon:
				action = env.action_space.sample()
			else:
				action, _ = agent.get_action(state_obs)

			next_state, reward, terminated, truncated, _ = env.step(
				action
			)

			reward = float(reward)
			done = bool(terminated or truncated)

			previous_steps = agent.steps

			loss = agent.update_weights(
				state=state_obs,
				action=int(action),
				reward=reward,
				next_state=next_state,
				done=done,
				discount_factor=config.discount_factor,
			)

			# Count only actual gradient updates, not replay warm-up steps.
			if agent.steps > previous_steps:
				state.loss_sum += float(loss)
				state.loss_count += 1

			total_reward += reward
			state_obs = next_state

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
					f"Satisfactory episodes: "
					f"{state.successful_episodes_window}/"
					f"{len(rewards)} | "
					f"Recent average reward: {rewards.mean():.2f} | "
					f"Window min: {rewards.min():.2f} | "
					f"Window max: {rewards.max():.2f} | "
					f"Average loss: {average_loss:.6f} | "
					f"{get_replay_description(agent, config)} | "
					f"Gradient updates: {agent.steps:,}"
				)

			state.loss_sum = 0.0
			state.loss_count = 0
			state.successful_episodes_window = 0

		if episode % config.save_interval == 0:
			save_model(config, agent, state)

			print(
				f"Checkpoint saved at episode {episode:,}: "
				f"{config.resolved_save_path(CURRENT_DIR)}"
			)

	return state


def print_config(
	config: Config,
	state: TrainingState,
) -> None:
	"""Print the effective training configuration."""

	epsilon_episodes = episode_to_reach_min_epsilon(
		config.min_epsilon,
		config.initial_epsilon,
		config.epsilon_decay,
	)

	replay_enabled = config.replay_batch_size > 0

	replay_description = (
		f"  Buffer capacity: {config.replay_capacity:,} transitions\n"
		f"  Batch size: {config.replay_batch_size:,}\n"
		f"  Warm-up: {config.replay_warmup:,} transitions\n"
		"  Training frequency: every environment step after warm-up"
		if replay_enabled
		else "  Experience replay: disabled\n"
		"  Training mode: online, one transition per update"
	)

	print(
		"Run\n"
		f"  Maximum episodes: {config.episodes:,}\n"
		f"  Device: {config.device}\n\n"
		"Model & optimizer\n"
		f"  Hidden dimensions: {config.hidden_dim}\n"
		f"  Learning rate: {config.learning_rate}\n"
		f"  Discount factor: {config.discount_factor}\n"
		f"  Target update frequency: "
		f"{config.target_update_freq} gradient updates\n\n"
		"Experience replay\n"
		f"{replay_description}\n\n"
		"Exploration\n"
		f"  Current epsilon: {state.epsilon:.5f}\n"
		f"  Minimum epsilon: {config.min_epsilon}\n"
		f"  Epsilon decay: {config.epsilon_decay}\n"
		f"  Episodes to reach minimum epsilon: "
		f"{epsilon_episodes:,}\n\n"
		"Logging & checkpoints\n"
		f"  Satisfactory reward threshold: "
		f"{config.satisfactory_reward_threshold}\n"
		f"  Log interval: {config.log_interval}\n"
		f"  Save interval: {config.save_interval}\n"
		f"  Save path: {config.resolved_save_path(CURRENT_DIR)}"
	)


def main() -> None:
	"""Load configuration, resume a checkpoint, and train."""

	args = parse_args()
	config = load_config(Config, args.config, CURRENT_DIR)

	if config is None:
		created_path = create_default_config(
			Config,
			args.config,
			CURRENT_DIR,
		)

		print(
			"Configuration file was not found. "
			f"Default configuration created at: {created_path}"
		)
		return

	env = gym.make("CartPole-v1")

	try:
		checkpoint: dict[str, Any] | None = None
		save_path = config.resolved_save_path(CURRENT_DIR)

		state = TrainingState(
			epsilon=config.initial_epsilon,
			recent_rewards=deque(maxlen=config.log_interval),
		)

		if not args.new_model and save_path.exists():
			print(f"Checking existing checkpoint: {save_path}")

			loaded = load_checkpoint(save_path, config.device)
			meta_data = loaded.get("meta", {})

			if not isinstance(meta_data, dict):
				raise ValueError(
					"Checkpoint metadata must be a dictionary."
				)

			saved_config = meta_data.get("config", {})

			if not isinstance(saved_config, dict):
				raise ValueError(
					"Checkpoint configuration must be a dictionary."
				)

			can_resume, mismatched_fields = config.can_resume_with(
				saved_config
			)

			if can_resume:
				config = config.merged_with_checkpoint(meta_data)
				checkpoint = loaded

				print(
					"  Compatible. Overwritable settings come from JSON; "
					"protected settings are restored from the checkpoint."
				)
			else:
				print(
					"  Critical configuration mismatch: "
					f"{mismatched_fields}\n"
					"  Starting a fresh model."
				)

		elif args.new_model:
			print("--new-model specified. Creating a fresh model.")

		else:
			print("No existing model found. Creating a fresh model.")

		agent = create_agent(env=env, config=config)

		if checkpoint is not None:
			agent.load_save_state(checkpoint)

			meta_data = checkpoint.get("meta", {})

			state.completed_episodes = int(
				meta_data.get("episodes", 0)
			)
			state.epsilon = float(
				meta_data.get("epsilon", config.initial_epsilon)
			)
			state.total_reward_sum = float(
				meta_data.get("total_reward_sum", 0.0)
			)
			state.total_successful_episodes = int(
				meta_data.get("total_successful_episodes", 0)
			)

			if agent.replay:
				replay_message = (
					f"restored {get_replay_size(agent):,} replay transitions"
				)
			else:
				replay_message = "experience replay is disabled"

			print(
				f"  Resuming from episode "
				f"{state.completed_episodes:,}; {replay_message}."
			)

		interrupted_by_user = False

		def interrupt_handler() -> None:
			nonlocal interrupted_by_user

			interrupted_by_user = True

			print(
				"\nTraining interrupted by user at episode "
				f"{state.completed_episodes:,}. "
				"No checkpoint was saved by the interrupt handler."
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
				f"Total satisfactory episodes "
				f"(>= {config.satisfactory_reward_threshold}): "
				f"{state.total_successful_episodes}"
			)
			print(f"Final epsilon: {state.epsilon:.5f}")
			print(f"Gradient updates: {agent.steps:,}")

			if agent.replay:
				print(
					f"Replay transitions retained: "
					f"{get_replay_size(agent):,}"
				)
			else:
				print("Experience replay: disabled")

			print(f"Model checkpoint saved to: {save_path}")

	finally:
		env.close()


if __name__ == "__main__":
	main()

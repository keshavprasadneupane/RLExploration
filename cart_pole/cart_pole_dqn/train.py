import argparse
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .agent import CartPoleDQNAgent

from shared.validate import (  
	bounded_float,
	bounded_int,
	resolve_path,
	validate_device,
)

CURRENT_DIR = Path(__file__).resolve().parent

DEFAULTS = {
	"episodes": 50_000_000,
	"hidden_dim": 8,
	"target_update_freq": 1000,
	"learning_rate": 0.001,
	"discount_factor": 0.99,
	"initial_epsilon": 1.0,
	"min_epsilon": 0.1,
	"epsilon_decay": 0.9999,
	"device": "cpu",
	"early_exit_reward": 500.0,
	"early_exit_count": 100,
	"log_interval": 10_000,
	"save_path": CURRENT_DIR / "models" / "cartpole_dqn.pth",
}


def create_agent(
	env: gym.Env,
	hidden_dim: int,
	target_update_freq: int,
	learning_rate: float,
	device: str,
) -> CartPoleDQNAgent:
	"""Create a CartPole DQN agent with the specified parameters."""

	if not isinstance(env.observation_space, gym.spaces.Box):
		raise ValueError(
			"Expected observation space to be of type Box."
		)

	if not isinstance(env.action_space, gym.spaces.Discrete):
		raise ValueError(
			"Expected action space to be of type Discrete."
		)

	return CartPoleDQNAgent(
		input_dim=env.observation_space.shape[0],
		output_dim=env.action_space.n,
		hidden_dim=hidden_dim,
		target_update_freq=target_update_freq,
		learning_rate=learning_rate,
		optimizer=torch.optim.Adam,
		criterion=None,
		device=device,
	)


def parse_args() -> argparse.Namespace:
	"""Parse and validate command-line arguments."""

	parser = argparse.ArgumentParser(
		description="Train a DQN agent on CartPole-v1."
	)

	parser.add_argument(
		"--episodes",
		type=bounded_int(1, 50_000_000),
		default=DEFAULTS["episodes"],
		help="Maximum number of training episodes.",
	)
	parser.add_argument(
		"--hidden-dim",
		type=bounded_int(1, 4096),
		default=DEFAULTS["hidden_dim"],
		help="Number of neurons in the hidden layer.",
	)
	parser.add_argument(
		"--target-update-freq",
		type=bounded_int(1, 10_000_000),
		default=DEFAULTS["target_update_freq"],
		help="Synchronize the target network every N updates.",
	)
	parser.add_argument(
		"--learning-rate",
		type=bounded_float(0.0, 1.0, min_inclusive=False),
		default=DEFAULTS["learning_rate"],
		help="Optimizer learning rate (0, 1].",
	)
	parser.add_argument(
		"--discount-factor",
		type=bounded_float(0.0, 1.0),
		default=DEFAULTS["discount_factor"],
		help="Discount factor for future rewards [0, 1].",
	)
	parser.add_argument(
		"--initial-epsilon",
		type=bounded_float(0.0, 1.0),
		default=DEFAULTS["initial_epsilon"],
		help="Initial exploration probability [0, 1].",
	)
	parser.add_argument(
		"--min-epsilon",
		type=bounded_float(0.0, 1.0),
		default=DEFAULTS["min_epsilon"],
		help="Minimum exploration probability [0, 1].",
	)
	parser.add_argument(
		"--epsilon-decay",
		type=bounded_float(0.0, 1.0, min_inclusive=False),
		default=DEFAULTS["epsilon_decay"],
		help="Epsilon decay multiplier (0, 1].",
	)
	parser.add_argument(
		"--device",
		type=validate_device,
		default=DEFAULTS["device"],
		help="PyTorch device: cpu, cuda, or mps.",
	)
	parser.add_argument(
		"--early-exit-reward",
		type=bounded_float(0.0, 500.0),
		default=DEFAULTS["early_exit_reward"],
		help="Reward threshold for counting a successful episode.",
	)
	parser.add_argument(
		"--early-exit-count",
		type=bounded_int(1, 1_000_000),
		default=DEFAULTS["early_exit_count"],
		help="Number of successful episodes required for early exit.",
	)
	parser.add_argument(
		"--log-interval",
		type=bounded_int(1, 10_000_000),
		default=DEFAULTS["log_interval"],
		help="Print training statistics every N episodes.",
	)
	parser.add_argument(
		"--save-path",
		type=resolve_path,
		default=DEFAULTS["save_path"],
		metavar="PATH",
		help="Path where the model checkpoint will be saved.",
	)

	args = parser.parse_args()

	if args.min_epsilon > args.initial_epsilon:
		parser.error(
			"--min-epsilon cannot exceed --initial-epsilon"
		)

	return args


def main() -> None:
	"""Train the DQN agent and save its checkpoint."""

	args = parse_args()
	env = gym.make("CartPole-v1")

	try:
		agent = create_agent(
			env=env,
			hidden_dim=args.hidden_dim,
			target_update_freq=args.target_update_freq,
			learning_rate=args.learning_rate,
			device=args.device,
		)

		epsilon = args.initial_epsilon
		recent_rewards: deque[float] = deque(maxlen=1000)
		total_reward_sum = 0.0
		successful_episodes = 0
		completed_episodes = 0

		loss_sum = 0.0
		loss_count = 0

		print(
			f"Training for up to {args.episodes:,} episodes\n"
			f"Hidden dimension: {args.hidden_dim}\n"
			f"Target update frequency: {args.target_update_freq}\n"
			f"Learning rate: {args.learning_rate}\n"
			f"Discount factor: {args.discount_factor}\n"
			f"Initial epsilon: {epsilon}\n"
			f"Minimum epsilon: {args.min_epsilon}\n"
			f"Epsilon decay: {args.epsilon_decay}\n"
			f"Device: {args.device}\n"
			f"Early-exit reward: {args.early_exit_reward}\n"
			f"Early-exit count: {args.early_exit_count}\n"
			f"Log interval: {args.log_interval}\n"
			f"Save path: {args.save_path}"
		)

		for episode in range(1, args.episodes + 1):
			state, _ = env.reset()

			terminated = False
			truncated = False
			total_reward = 0.0

			while not (terminated or truncated):
				# Epsilon-greedy action selection.
				if random.random() < epsilon:
					action = env.action_space.sample()
				else:
					action, _ = agent.get_action(state)

				(
					next_state,
					reward,
					terminated,
					truncated,
					_,
				) = env.step(action)

				reward = float(reward)

				# Bootstrap through time-limit truncations, but not
				# through genuine terminal states.
				loss = agent.update_weights(
					state=state,
					action=action,
					reward=reward,
					next_state=next_state,
					done=terminated,
					discount_factor=args.discount_factor,
				)

				loss_sum += loss
				loss_count += 1

				state = next_state
				total_reward += reward

			completed_episodes += 1
			total_reward_sum += total_reward
			recent_rewards.append(total_reward)

			if total_reward >= args.early_exit_reward:
				successful_episodes += 1

			epsilon = max(
				args.min_epsilon,
				epsilon * args.epsilon_decay,
			)

			if episode % args.log_interval == 0:
				average_loss = (
					loss_sum / loss_count
					if loss_count > 0
					else 0.0
				)

				print(
					f"Episode {episode:,} | "
					f"Epsilon: {epsilon:.5f} | "
					f"Recent average reward: "
					f"{np.mean(recent_rewards):.2f} | "
					f"Min: {min(recent_rewards):.2f} | "
					f"Max: {max(recent_rewards):.2f} | "
					f"Average loss: {average_loss:.6f}"
				)

				loss_sum = 0.0
				loss_count = 0

			if total_reward >= args.early_exit_reward:
				print(
					f"Episode {episode:,}: "
					f"reward={total_reward:.1f}, "
					f"successful episodes={successful_episodes}"
				)

			if successful_episodes >= args.early_exit_count:
				print(
					f"Early exit at episode {episode:,}: "
					f"{successful_episodes} episodes reached "
					f"the reward threshold of "
					f"{args.early_exit_reward}."
				)
				break

		save_path: Path = args.save_path
		save_path.parent.mkdir(parents=True, exist_ok=True)

		save_state = agent.get_save_state()
		save_state.update(
			{
				"episodes": completed_episodes,
				"epsilon": epsilon,
				"steps": agent.steps,
				"config": {
					key: str(value) if isinstance(value, Path) else value
					for key, value in vars(args).items()
				},
			}
		)
		torch.save(
			save_state,
			save_path,
		)

		print(
			f"\nTraining completed: "
			f"{completed_episodes:,} episodes"
		)
		print(
			f"Overall average reward: "
			f"{total_reward_sum / completed_episodes:.2f}"
		)
		print(f"Successful episodes: {successful_episodes}")
		print(f"Final epsilon: {epsilon:.5f}")
		print(f"Model checkpoint saved to: {save_path}")

	finally:
		env.close()


if __name__ == "__main__":
	main()

import argparse
import pickle
from pathlib import Path
from statistics import multimode
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from shared.agent.dqn_agent import DQNetworkTestAgent
from shared.network.feed_forward import FeedForward
from shared.wrapper import stop_by_user
from shared.validate import (
	bounded_int,
	resolve_path,
	validate_device,
)

CURRENT_DIR = Path(__file__).resolve().parent

DEFAULTS = {
	"model_path": CURRENT_DIR / "models" / "mountain_car_dqn.pth",
	"episodes": 100,
	"render_mode": "none",
	"device": "cpu",
}


def parse_args() -> argparse.Namespace:
	"""Parse and validate command-line arguments."""

	parser = argparse.ArgumentParser(
		description="Evaluate a trained DQN agent on MountainCar-v0."
	)

	parser.add_argument(
		"--model-path",
		type=lambda path: resolve_path(CURRENT_DIR, path),
		default=DEFAULTS["model_path"],
		metavar="PATH",
		help="Path to the saved DQN checkpoint.",
	)

	parser.add_argument(
		"--episodes",
		type=bounded_int(1, 1_000_000),
		default=DEFAULTS["episodes"],
		help="Number of evaluation episodes in headless mode.",
	)

	parser.add_argument(
		"--render-mode",
		choices=("human", "none"),
		default=DEFAULTS["render_mode"],
		help=(
			"'human' renders the environment continuously; "
			"'none' evaluates a fixed number of episodes."
		),
	)

	parser.add_argument(
		"--device",
		type=validate_device,
		default=DEFAULTS["device"],
		help="PyTorch device: cpu, cuda, or mps.",
	)

	parser.add_argument(
		"-v",
		"--verbose",
		action="store_true",
		help="Print individual episode rewards during evaluation.",
	)

	return parser.parse_args()


def load_checkpoint(
	model_path: Path,
	device: str,
) -> dict[str, Any]:
	"""Load and validate a saved DQN checkpoint."""

	if not model_path.is_file():
		raise SystemExit(
			f"Error: model checkpoint not found: {model_path}"
		)

	try:
		checkpoint = torch.load(
			model_path,
			map_location=device,
			weights_only=False,  # Load the entire checkpoint, including metadata
		)
	except pickle.UnpicklingError as error:
		raise SystemExit(
			"Error: the checkpoint contains objects that safe loading "
			"rejects (for example numpy scalars in its 'meta' block). "
			"Resume training once with the current train.py so the "
			"checkpoint is re-saved with plain Python types.\n"
			f"Details: {error}"
		) from error

	if not isinstance(checkpoint, dict):
		raise ValueError(
			"Invalid checkpoint: expected a dictionary."
		)

	if "network_state_dict" not in checkpoint:
		raise ValueError(
			"Invalid checkpoint: missing 'network_state_dict'."
		)

	meta = checkpoint.get("meta")
	if meta is not None and not isinstance(meta, dict):
		raise ValueError(
			"Invalid checkpoint: 'meta' must be a dictionary."
		)

	return checkpoint


def get_meta(checkpoint: dict[str, Any]) -> dict[str, Any]:
	"""Return the checkpoint's training metadata, or {} for older checkpoints."""

	return checkpoint.get("meta") or {}


def infer_network_dims(checkpoint: dict[str, Any]) -> tuple[list[int], int]:
	"""Automatically infer hidden dimensions list and output dimension from saved weights."""

	network_state = checkpoint["network_state_dict"]
	
	# Weight keys in FeedForward are typically sequential (e.g., nn.0.weight, nn.2.weight, etc.)
	weight_keys = [k for k in network_state.keys() if k.endswith(".weight")]
	
	if not weight_keys:
		raise ValueError("Cannot infer dimensions: no weight tensors found in state_dict.")

	weight_keys.sort(key=lambda x: int(x.split(".")[1]))

	# All weights except the last one belong to hidden layers
	hidden_dims = [int(network_state[k].shape[0]) for k in weight_keys[:-1]]
	output_dim = int(network_state[weight_keys[-1]].shape[0])

	if not hidden_dims or output_dim < 1:
		raise ValueError("Invalid checkpoint: dimensions must be positive integers.")

	return hidden_dims, output_dim


def resolve_network_dims(checkpoint: dict[str, Any]) -> tuple[list[int], int]:
	"""
	Get the network dimensions from the saved weights.

	The weights are the source of truth. If the checkpoint also stores the
	training config (meta["config"]), its hidden_dim must agree with them.
	"""

	hidden_dims, output_dim = infer_network_dims(checkpoint)

	saved_config = get_meta(checkpoint).get("config", {})
	saved_hidden_dim = (
		saved_config.get("hidden_dim")
		if isinstance(saved_config, dict)
		else None
	)

	if saved_hidden_dim is not None:
		# Normalize saved_hidden_dim to a list for comparison
		expected_dims = (
			saved_hidden_dim
			if isinstance(saved_hidden_dim, list)
			else [saved_hidden_dim]
		)
		if expected_dims != hidden_dims:
			raise ValueError(
				"Checkpoint is inconsistent: saved config says "
				f"hidden_dim={saved_hidden_dim}, but the weights have "
				f"hidden_dim={hidden_dims}."
			)

	return hidden_dims, output_dim


def print_training_info(checkpoint: dict[str, Any]) -> None:
	"""Print training progress stored in the checkpoint, if present."""

	meta = get_meta(checkpoint)

	if not meta:
		print("Training info: not available (checkpoint has no metadata)")
		return

	epsilon = meta.get("epsilon")
	epsilon_text = f"{epsilon:.5f}" if epsilon is not None else "n/a"

	print(f"Trained episodes: {meta.get('episodes', 0):,}")
	print(f"Training steps: {meta.get('steps', 0):,}")
	print(f"Final epsilon: {epsilon_text}")


def run_episode(
	env: gym.Env,
	agent: DQNetworkTestAgent,
) -> float:
	"""Run one evaluation episode using greedy actions."""

	state, _ = env.reset()

	terminated = False
	truncated = False
	total_reward = 0.0

	while not (terminated or truncated):
		action_idx, _ = agent.get_action(state)

		next_state, reward, terminated, truncated, _ = env.step(action_idx)

		state = next_state
		total_reward += float(reward)

	return total_reward


def print_statistics(rewards: list[float]) -> None:
	"""Print descriptive statistics for episode rewards."""

	values = np.asarray(rewards, dtype=np.float64)

	print("\nEvaluation statistics")
	print("=" * 40)
	print(f"Episodes:           {len(values)}")
	print(f"Mean:               {np.mean(values):.2f}")
	print(f"Median:             {np.median(values):.2f}")
	print(f"Minimum:            {np.min(values):.2f}")
	print(f"Maximum:            {np.max(values):.2f}")
	print(f"Range:              {np.ptp(values):.2f}")
	print(f"Standard deviation: {np.std(values):.2f}")
	print(f"25th percentile:    {np.percentile(values, 25):.2f}")
	print(f"75th percentile:    {np.percentile(values, 75):.2f}")
	print(f"95th percentile:    {np.percentile(values, 95):.2f}")
	print(f"Rewards >= -110:    {np.count_nonzero(values >= -110.0)}")
	print(f"Success rate (>=-110): {np.mean(values >= -110.0) * 100:.2f}%")


@stop_by_user(message="Evaluation interrupted by user.")
def main() -> None:
	"""Load a trained DQN checkpoint and evaluate the agent on MountainCar-v0."""

	args = parse_args()
	model_path: Path = args.model_path

	# Load checkpoint and automatically grab dimensions
	checkpoint = load_checkpoint(model_path, args.device)
	hidden_dims, output_dim = resolve_network_dims(checkpoint)

	env = gym.make(
		"MountainCar-v0",
		render_mode=(
			"human" if args.render_mode == "human" else None
		),
	)

	try:
		if not isinstance(env.observation_space, gym.spaces.Box):
			raise ValueError(
				"Expected observation space to be of type Box."
			)

		if not isinstance(env.action_space, gym.spaces.Discrete):
			raise ValueError(
				"Expected action space to be of type Discrete."
			)

		# Instantiate the network with the full hidden_dims list
		network = FeedForward(
			input_dim=env.observation_space.shape[0],
			output_dim=output_dim,
			hidden_dim=hidden_dims,
			device=args.device,
		)

		agent = DQNetworkTestAgent(
			network=network,
		)

		agent.network.load_state_dict(checkpoint["network_state_dict"])
		agent.network.eval()

		print(f"Model checkpoint: {model_path}")
		print_training_info(checkpoint)
		print()
		print(f"Hidden dimensions: {hidden_dims}")
		print(f"Output dimension: {output_dim}")
		print(f"Device: {args.device}")
		print(f"Render mode: {args.render_mode}")

		if args.render_mode == "human":
			print(
				"Rendering continuously. Close the window "
				"or press Ctrl+C to stop."
			)

			episode = 0

			while True:
				episode += 1
				reward = run_episode(env, agent)

				if args.verbose:
					print(f"Episode {episode}: reward={reward:.2f}")

		else:
			print(f"Evaluating for {args.episodes:,} episodes.")

			rewards: list[float] = []

			for episode in range(1, args.episodes + 1):
				reward = run_episode(env, agent)
				rewards.append(reward)

				if args.verbose:
					print(
						f"Episode {episode:>4}/{args.episodes} | "
						f"Reward: {reward:.2f}"
					)

			print_statistics(rewards)
	finally:
		env.close()


if __name__ == "__main__":
	main()
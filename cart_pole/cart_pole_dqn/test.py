
import argparse
from pathlib import Path
from statistics import multimode
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .agent import CartPoleDQNAgent
from shared.validate import (
    bounded_int,
    resolve_path,
    validate_device,
)

CURRENT_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "model_path": CURRENT_DIR / "models" / "cartpole_dqn.pth",
    "episodes": 100,
    "render_mode": "none",
    "device": "cpu",
}


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained CartPole DQN agent."
    )

    parser.add_argument(
        "--model-path",
        type=resolve_path,
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

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Invalid checkpoint: expected a dictionary."
        )

    if "network_state_dict" not in checkpoint:
        raise ValueError(
            "Invalid checkpoint: missing 'network_state_dict'."
        )

    if "target_network_state_dict" not in checkpoint:
        raise ValueError(
            "Invalid checkpoint: missing 'target_network_state_dict'."
        )

    return checkpoint


def get_hidden_dim(checkpoint: dict[str, Any]) -> int:
    """Infer the hidden layer size from the saved network weights."""

    network_state = checkpoint["network_state_dict"]
    weight_key = "nn.0.weight"

    if weight_key not in network_state:
        raise ValueError(
            f"Cannot infer hidden dimension: "
            f"checkpoint is missing '{weight_key}'."
        )

    hidden_dim = network_state[weight_key].shape[0]

    if hidden_dim < 1:
        raise ValueError(
            "Invalid checkpoint: hidden dimension must be positive."
        )

    return int(hidden_dim)


def run_episode(
    env: gym.Env,
    agent: CartPoleDQNAgent,
) -> float:
    """Run one evaluation episode using greedy actions."""

    state, _ = env.reset()

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = agent.get_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        state = next_state
        total_reward += float(reward)

    return total_reward


def print_statistics(rewards: list[float]) -> None:
    """Print descriptive statistics for episode rewards."""

    values = np.asarray(rewards, dtype=np.float64)
    modes = multimode(rewards)

    print("\nEvaluation statistics")
    print("=" * 40)
    print(f"Episodes:           {len(values)}")
    print(f"Mean:               {np.mean(values):.2f}")
    print(f"Median:             {np.median(values):.2f}")
    print(
        f"Mode:               "
        f"{', '.join(f'{mode:.0f}' for mode in modes)}"
    )
    print(f"Minimum:            {np.min(values):.2f}")
    print(f"Maximum:            {np.max(values):.2f}")
    print(f"Range:              {np.ptp(values):.2f}")
    print(f"Standard deviation: {np.std(values):.2f}")
    print(f"25th percentile:    {np.percentile(values, 25):.2f}")
    print(f"75th percentile:    {np.percentile(values, 75):.2f}")
    print(f"95th percentile:    {np.percentile(values, 95):.2f}")
    print(f"Rewards >= 475:     {np.count_nonzero(values >= 475)}")
    print(f"Success rate:       {np.mean(values >= 475) * 100:.2f}%")


def main() -> None:
    """Load a trained DQN checkpoint and evaluate the agent."""

    args = parse_args()
    model_path: Path = args.model_path

    # Load the checkpoint before constructing the agent.
    checkpoint = load_checkpoint(model_path, args.device)
    hidden_dim = get_hidden_dim(checkpoint)

    env = gym.make(
        "CartPole-v1",
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

        # Match the architecture used to create the checkpoint.
        agent = CartPoleDQNAgent(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.n,
            hidden_dim=hidden_dim,
            device=args.device,
        )

        agent.load_save_state(checkpoint)

        agent.network.eval()
        agent.target_network.eval()

        print(f"Model checkpoint: {model_path}")
        print(f"Hidden dimension: {hidden_dim}")
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
                    print(f"Episode {episode}: reward={reward:.0f}")

        else:
            print(f"Evaluating for {args.episodes:,} episodes.")

            rewards: list[float] = []

            for episode in range(1, args.episodes + 1):
                reward = run_episode(env, agent)
                rewards.append(reward)

                if args.verbose:
                    print(
                        f"Episode {episode:>4}/{args.episodes} | "
                        f"Reward: {reward:.0f}"
                    )

            print_statistics(rewards)

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()

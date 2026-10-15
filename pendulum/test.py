import argparse
from pathlib import Path
from statistics import multimode
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from shared.agent.simple_dqn_agent import SimpleDQNTestAgent
from shared.mapper.dicrete_mapper import DiscreteActionMapper
from shared.network.feed_forward import FeedForward
from shared.wrapper import stop_by_user
from shared.validate import (
    bounded_int,
    resolve_path,
    validate_device,
)

CURRENT_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "model_path": CURRENT_DIR / "models" / "pendulum_dqn_72_h.pth",
    "episodes": 100,
    "render_mode": "none",
    "device": "cpu",
}


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on Pendulum-v1."
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

    return checkpoint


def infer_network_dims(checkpoint: dict[str, Any]) -> tuple[int, int]:
    """Automatically infer hidden dimension and output dimension from saved weights."""

    network_state = checkpoint["network_state_dict"]
    
    # Typically, Sequential linear layers in FeedForward are keyed like nn.0.weight, nn.2.weight, etc.
    # Find all weight keys to reliably locate the first layer (input -> hidden) and last layer (hidden -> output).
    weight_keys = [k for k in network_state.keys() if k.endswith(".weight")]
    
    if not weight_keys:
        raise ValueError("Cannot infer dimensions: no weight tensors found in state_dict.")

    # Sort keys by layer index (e.g., nn.0.weight, nn.2.weight)
    weight_keys.sort(key=lambda x: int(x.split(".")[1]))

    first_weight = network_state[weight_keys[0]]
    last_weight = network_state[weight_keys[-1]]

    # first_weight shape: [hidden_dim, input_dim] -> rows are hidden_dim
    hidden_dim = int(first_weight.shape[0])
    
    # last_weight shape: [output_dim, hidden_dim] -> rows are output_dim
    output_dim = int(last_weight.shape[0])

    if hidden_dim < 1 or output_dim < 1:
        raise ValueError("Invalid checkpoint: dimensions must be positive integers.")

    return hidden_dim, output_dim


def run_episode(
    env: gym.Env,
    agent: SimpleDQNTestAgent,
    action_mapper: DiscreteActionMapper,
) -> float:
    """Run one evaluation episode using greedy actions mapped to continuous space."""

    state, _ = env.reset()

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action_idx, _ = agent.get_action(state)
        continuous_action = action_mapper.to_continuous(action_idx)

        next_state, reward, terminated, truncated, _ = env.step(continuous_action)

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
        f"{', '.join(f'{mode:.2f}' for mode in modes)}"
    )
    print(f"Minimum:            {np.min(values):.2f}")
    print(f"Maximum:            {np.max(values):.2f}")
    print(f"Range:              {np.ptp(values):.2f}")
    print(f"Standard deviation: {np.std(values):.2f}")
    print(f"25th percentile:    {np.percentile(values, 25):.2f}")
    print(f"75th percentile:    {np.percentile(values, 75):.2f}")
    print(f"95th percentile:    {np.percentile(values, 95):.2f}")
    print(f"Rewards >= -200:    {np.count_nonzero(values >= -200.0)}")
    print(f"Success rate (>-200): {np.mean(values >= -200.0) * 100:.2f}%")


@stop_by_user(message="Evaluation interrupted by user.")
def main() -> None:
    """Load a trained DQN checkpoint and evaluate the agent on Pendulum-v1."""

    args = parse_args()
    model_path: Path = args.model_path

    # Load checkpoint and automatically grab dimensions
    checkpoint = load_checkpoint(model_path, args.device)
    hidden_dim, output_dim = infer_network_dims(checkpoint)

    env = gym.make(
        "Pendulum-v1",
        render_mode=(
            "human" if args.render_mode == "human" else None
        ),
    )

    try:
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                "Expected observation space to be of type Box."
            )

        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                "Expected action space to be of type Box."
            )

        # Initialize the action mapper with the dynamically inferred output dimension
        action_mapper = DiscreteActionMapper(env.action_space, num_actions=output_dim)

        # Instantiate the network with dynamically grabbed dimensions
        network = FeedForward(
            input_dim=env.observation_space.shape[0],
            output_dim=output_dim,
            hidden_dim=[hidden_dim],
            device=args.device,
        )

        agent = SimpleDQNTestAgent(
            network=network,
        )

        agent.network.load_state_dict(checkpoint["network_state_dict"])
        agent.network.eval()

        print(f"Model checkpoint: {model_path}")
        print(f"Hidden dimension: {hidden_dim}")
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
                reward = run_episode(env, agent, action_mapper)

                if args.verbose:
                    print(f"Episode {episode}: reward={reward:.2f}")

        else:
            print(f"Evaluating for {args.episodes:,} episodes.")

            rewards: list[float] = []

            for episode in range(1, args.episodes + 1):
                reward = run_episode(env, agent, action_mapper)
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
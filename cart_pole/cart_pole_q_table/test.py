import argparse
from pathlib import Path
from statistics import multimode

import gymnasium as gym
import numpy as np

from .agent import CartPoleQLearningAgent


CURRENT_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "q_table_path": CURRENT_DIR /"models"/ "cartpole_q_table.npy",
    "episodes": 100,
    "render_mode": "none",
}


def bounded_int(minimum: int, maximum: int):
    def validate(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "must be an integer"
            ) from exc

        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be between {minimum} and {maximum}"
            )

        return number

    return validate


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the current working directory."""
    return Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CartPole Q-learning agent.",
    )

    parser.add_argument(
        "--q-table-path",
        type=resolve_path,
        default=DEFAULTS["q_table_path"],
        metavar="PATH",
        help="Path to the saved Q-table.",
    )

    parser.add_argument(
        "--episodes",
        type=bounded_int(1, 1_000_000),
        default=DEFAULTS["episodes"],
        help="Number of evaluation episodes when render mode is none.",
    )

    parser.add_argument(
        "--render-mode",
        choices=("human", "none"),
        default=DEFAULTS["render_mode"],
        help=(
            "'human' continuously renders the agent; "
            "'none' evaluates a fixed number of episodes."
        ),
    )

    return parser.parse_args()


def run_episode(
    env: gym.Env,
    agent: CartPoleQLearningAgent,
) -> float:
    state, _ = env.reset()
    state = agent.discretize_state(state)

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action = agent.choose_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        state = agent.discretize_state(next_state)
        total_reward += float(reward)

    return total_reward


def print_statistics(rewards: list[float]) -> None:
    values = np.asarray(rewards, dtype=np.float64)
    modes = multimode(rewards)

    print("\nEvaluation statistics")
    print("=" * 40)
    print(f"Episodes:          {len(values)}")
    print(f"Mean:              {np.mean(values):.2f}")
    print(f"Median:            {np.median(values):.2f}")
    print(
        f"Mode:              "
        f"{', '.join(f'{mode:.0f}' for mode in modes)}"
    )
    print(f"Minimum:           {np.min(values):.2f}")
    print(f"Maximum:           {np.max(values):.2f}")
    print(f"Range:             {np.ptp(values):.2f}")
    print(f"Standard deviation:{np.std(values):>8.2f}")
    print(f"25th percentile:   {np.percentile(values, 25):.2f}")
    print(f"75th percentile:   {np.percentile(values, 75):.2f}")
    print(f"95th percentile:   {np.percentile(values, 95):.2f}")
    print(f"Rewards >= 475:    {np.count_nonzero(values >= 475)}")
    print(f"Success rate:      {np.mean(values >= 475) * 100:.2f}%")


def main() -> None:
    args = parse_args()

    q_table_path: Path = args.q_table_path

    if not q_table_path.is_file():
        raise SystemExit(
            f"Error: Q-table file not found: {q_table_path}"
        )

    env = gym.make(
        "CartPole-v1",
        render_mode=(
            "human" if args.render_mode == "human" else None
        ),
    )

    try:
        agent = CartPoleQLearningAgent(env=env)
        agent.load_q_table(q_table_path)

        # Greedy evaluation: always choose the best learned action.
        agent.epsilon = 0.0

        print(f"Q-table: {q_table_path}")
        print(f"Render mode: {args.render_mode}")

        if args.render_mode == "human":
            print("Rendering continuously. Close the window or press Ctrl+C to stop.")

            episode = 0

            while True:
                episode += 1
                reward = run_episode(env, agent)
                print(f"Episode {episode}: reward={reward:.0f}")

        else:
            print(f"Evaluating for {args.episodes:,} episodes.")

            rewards: list[float] = []

            for episode in range(1, args.episodes + 1):
                reward = run_episode(env, agent)
                rewards.append(reward)

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

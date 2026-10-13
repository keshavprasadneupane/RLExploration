import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np

from .agent import CartPoleQLearningAgent


from shared.validate import (
	bounded_float,
	bounded_int,
	resolve_path,
)

CURRENT_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "episodes": 50_000_000,
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "initial_epsilon": 1.0,
    "min_epsilon": 0.05,
    "epsilon_decay": 0.9999,
    "early_exit_reward": 500.0,
    "early_exit_count": 100,
    "log_interval": 10_000,
    "save_path": CURRENT_DIR / "models" / "cartpole_q_table.npy",
}


# def bounded_int(minimum: int, maximum: int):
#     def validate(value: str) -> int:
#         try:
#             number = int(value)
#         except ValueError as exc:
#             raise argparse.ArgumentTypeError(
#                 "must be an integer"
#             ) from exc

#         if not minimum <= number <= maximum:
#             raise argparse.ArgumentTypeError(
#                 f"must be between {minimum} and {maximum}"
#             )

#         return number

#     return validate


# def bounded_float(
#     minimum: float,
#     maximum: float,
#     *,
#     min_inclusive: bool = True,
# ):
#     def validate(value: str) -> float:
#         try:
#             number = float(value)
#         except ValueError as exc:
#             raise argparse.ArgumentTypeError(
#                 "must be a number"
#             ) from exc

#         if not np.isfinite(number):
#             raise argparse.ArgumentTypeError(
#                 "must be a finite number"
#             )

#         valid_min = (
#             number >= minimum
#             if min_inclusive
#             else number > minimum
#         )

#         if not valid_min or number > maximum:
#             left = "[" if min_inclusive else "("
#             raise argparse.ArgumentTypeError(
#                 f"must be in the range "
#                 f"{left}{minimum}, {maximum}]"
#             )

#         return number

#     return validate


# def resolve_path(path: str | Path) -> Path:
#     """Resolve a path relative to the current working directory."""
#     return Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Q-learning agent on CartPole-v1.")

    parser.add_argument(
        "--episodes",
        type=bounded_int(1, 50_000_000),
        default=DEFAULTS["episodes"],
        help="Maximum number of training episodes (1 to 50,000,000).",
    )
    parser.add_argument(
        "--learning-rate",
        type=bounded_float(0.0, 1.0, min_inclusive=False),
        default=DEFAULTS["learning_rate"],
        help="Learning rate (0, 1].",
    )
    parser.add_argument(
        "--discount-factor",
        type=bounded_float(0.0, 1.0),
        default=DEFAULTS["discount_factor"],
        help="Discount factor [0, 1].",
    )
    parser.add_argument(
        "--initial-epsilon",
        type=bounded_float(0.0, 1.0),
        default=DEFAULTS["initial_epsilon"],
        help="Initial exploration rate [0, 1].",
    )
    parser.add_argument(
        "--min-epsilon",
        type=bounded_float(0.0, 1.0),
        default=DEFAULTS["min_epsilon"],
        help="Minimum exploration rate [0, 1].",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=bounded_float(0.0, 1.0, min_inclusive=False),
        default=DEFAULTS["epsilon_decay"],
        help="Epsilon decay factor (0, 1].",
    )
    parser.add_argument(
        "--early-exit-reward",
        type=bounded_float(0.0, 500.0),
        default=DEFAULTS["early_exit_reward"],
        help="Reward threshold for counting a successful episode [0, 500].",
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
        type=lambda path: resolve_path(CURRENT_DIR, path),
        default=DEFAULTS["save_path"],
        metavar="PATH",
        help=(
            "Path where the Q-table will be saved. "
            "Use Relative or absolute paths. "
        ),
    )
    parser.add_argument(
        "--show-env",
        action="store_true",
        help="Render the environment visually.",
    )

    args = parser.parse_args()

    if args.min_epsilon > args.initial_epsilon:
        parser.error(
            "--min-epsilon cannot exceed --initial-epsilon"
        )

    return args


def main() -> None:
    args = parse_args()

    env = gym.make(
        "CartPole-v1",
        render_mode="human" if args.show_env else None,
    )

    try:
        agent = CartPoleQLearningAgent(
            env=env,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            initial_epsilon=args.initial_epsilon,
            min_epsilon=args.min_epsilon,
            epsilon_decay=args.epsilon_decay,
        )

        print(
            f"Training for up to {args.episodes:,} episodes\n"
            f"Learning rate: {agent.learning_rate}\n"
            f"Discount factor: {agent.discount_factor}\n"
            f"Initial epsilon: {agent.epsilon}\n"
            f"Minimum epsilon: {agent.min_epsilon}\n"
            f"Epsilon decay: {agent.epsilon_decay}\n"
            f"Early-exit reward: {args.early_exit_reward}\n"
            f"Early-exit count: {args.early_exit_count}\n"
            f"Log interval: {args.log_interval}\n"
            f"Rendering: {args.show_env}\n"
            f"Save path: {args.save_path}"
        )

        recent_rewards: deque[float] = deque(maxlen=1000)
        total_reward_sum = 0.0
        successful_episodes = 0
        completed_episodes = 0

        for episode in range(1, args.episodes + 1):
            state, _ = env.reset()
            state = agent.discretize_state(state)

            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action = agent.choose_action(state)

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                next_state = agent.discretize_state(next_state)
                reward = float(reward)

                agent.update_q_value(
                    state,
                    action,
                    reward,
                    terminated,
                    next_state,
                )

                state = next_state
                total_reward += reward

            completed_episodes += 1
            total_reward_sum += total_reward
            recent_rewards.append(total_reward)

            if total_reward >= args.early_exit_reward:
                successful_episodes += 1

                print(
                    f"Episode {episode:,}: "
                    f"reward={total_reward:.1f}, "
                    f"successful episodes={successful_episodes}"
                )

            agent.decay_epsilon()

            if episode % args.log_interval == 0:
                print(
                    f"Episode {episode:,} | "
                    f"Epsilon: {agent.epsilon:.5f} | "
                    f"Recent average: {np.mean(recent_rewards):.2f} | "
                    f"Min: {min(recent_rewards):.2f} | "
                    f"Max: {max(recent_rewards):.2f}"
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

        agent.save_q_table(save_path)

        print(
            f"\nTraining completed: "
            f"{completed_episodes:,} episodes"
        )
        print(
            f"Overall average reward: "
            f"{total_reward_sum / completed_episodes:.2f}"
        )
        print(f"Successful episodes: {successful_episodes}")
        print(f"Q-table saved to: {save_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
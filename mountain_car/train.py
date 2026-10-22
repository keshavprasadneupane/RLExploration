import shared.tools.execution_timer  # noqa: F401 — automatic timer

import argparse
import dataclasses
import json
import pickle
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from shared.agent.dqn_agent import DQNetworkTrainAgent
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
        raise ValueError(f"Configuration must contain a JSON object: {resolved_path}")

    valid_keys = {item.name for item in dataclasses.fields(Config)}
    filtered_data = {key: value for key, value in data.items() if key in valid_keys}
    return Config(**filtered_data)


def episode_to_reach_min_epsilon(y: float, x: float, a: float) -> int:
    """Calculate episodes needed to reach minimum epsilon."""
    if not 0 < x <= y <= 1 or not 0 < a < 1:
        return 0
    return int(np.ceil(np.log(y / x) / np.log(a)))


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
        description="Train a replay-based DQN agent on MountainCar-v0."
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


def create_agent(env: gym.Env, config: Config) -> DQNetworkTrainAgent:
    """Create the DQN network, optimizer, and replay buffer."""
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
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

    return DQNetworkTrainAgent(
        network=network,
        optimizer=optimizer,
        target_update_freq=config.target_update_freq,
        replay_capacity=config.replay_capacity,
        batch_size=config.replay_batch_size,
    )


def to_safe_checkpoint_value(value: Any) -> Any:
    """
    Convert NumPy values in replay memory and metadata to safe primitive values.

    This allows checkpoints to be loaded with torch.load(weights_only=True).
    Tensors and ordinary Python values are retained.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_safe_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [to_safe_checkpoint_value(item) for item in value]
    return value


def save_model(
    config: Config,
    agent: DQNetworkTrainAgent,
    state: TrainingState,
) -> None:
    """Save model, optimizer, replay memory, progress, and configuration."""
    save_path = config.resolved_save_path(CURRENT_DIR)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_state = agent.get_save_state()
    save_state["meta"] = {
        "episodes": int(state.completed_episodes),
        "epsilon": float(state.epsilon),
        "steps": int(agent.steps),
        "total_reward_sum": float(state.total_reward_sum),
        "total_successful_episodes": int(state.total_successful_episodes),
        "config": asdict(config),
    }
    torch.save(to_safe_checkpoint_value(save_state), save_path)


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    """Load a checkpoint safely, with a compatibility fallback for trusted legacy files."""
    try:
        loaded = torch.load(path, map_location=device, weights_only=True)
    except (RuntimeError, pickle.UnpicklingError, AttributeError, TypeError):
        # Compatibility fallback for older locally-created checkpoints that
        # contain NumPy objects. Only use this with a checkpoint you trust.
        print(
            "Safe checkpoint loading failed; retrying legacy loading. "
            "Only load checkpoints from a trusted source."
        )
        loaded = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(loaded, dict):
        raise ValueError("Invalid checkpoint format.")
    return loaded


def calculate_mountain_car_reward(
    position: float,
    velocity: float,
    midpoint: float,
    goal_position: float = 0.5,
) -> float:
    """Reward reaching the goal and encourage progress on the right-hand slope."""
    if position >= goal_position:
        return 20.0

    reward = -1.0
    absolute_velocity = abs(velocity)
    if position > midpoint:
        reward = (
            (-1.0 / (midpoint - goal_position)) * (position - goal_position)
            + absolute_velocity * 4.0
        )
    return float(reward)


def train_loop(
    config: Config,
    env: gym.Env,
    agent: DQNetworkTrainAgent,
    state: TrainingState,
) -> TrainingState:
    """Train up to the configured maximum episode count."""
    observation_space = cast(gym.spaces.Box, env.observation_space)
    midpoint = float(
        observation_space.low[0]
        + (observation_space.high[0] - observation_space.low[0]) / 2
    )

    if state.recent_rewards.maxlen != config.log_interval:
        state.recent_rewards = deque(
            state.recent_rewards, maxlen=config.log_interval
        )

    for episode in range(state.completed_episodes + 1, config.episodes + 1):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            if random.random() < state.epsilon:
                action = env.action_space.sample()
            else:
                action, _ = agent.get_action(observation)

            next_observation, _, terminated, truncated, _ = env.step(action)
            reward = calculate_mountain_car_reward(
                float(next_observation[0]),
                float(next_observation[1]),
                midpoint=midpoint,
            )

            loss = agent.update_weights(
                state=observation,
                action=int(action),
                reward=reward,
                next_state=next_observation,
                # Truncation is a time limit, not a true terminal state.
                done=bool(terminated),
                discount_factor=config.discount_factor,
            )

            # The agent returns 0.0 before replay is large enough to train.
            # Do not include those warm-up calls in the reported loss average.
            if len(agent.replay_buffer) >= agent.batch_size:
                state.loss_sum += float(loss)
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
            rewards = np.asarray(state.recent_rewards, dtype=np.float64)
            average_loss = (
                state.loss_sum / state.loss_count if state.loss_count else 0.0
            )

            if rewards.size:
                print(
                    f"Episode {episode:,} | "
                    f"Epsilon: {state.epsilon:.5f} | "
                    f"Satisfactory episodes: "
                    f"{state.successful_episodes_window}/{len(rewards)} | "
                    f"Recent average reward: {rewards.mean():.2f} | "
                    f"Window min: {rewards.min():.2f} | "
                    f"Window max: {rewards.max():.2f} | "
                    f"Average loss: {average_loss:.6f} | "
                    f"Replay: {len(agent.replay_buffer):,}/"
                    f"{config.replay_capacity:,} | "
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


def print_config(config: Config, state: TrainingState) -> None:
    """Print the effective training and replay configuration."""
    epsilon_episodes = episode_to_reach_min_epsilon(
        config.min_epsilon,
        config.initial_epsilon,
        config.epsilon_decay,
    )
    print(
        "Run\n"
        f"  Maximum episodes: {config.episodes:,}\n"
        f"  Device: {config.device}\n\n"
        "Model & optimizer\n"
        f"  Hidden dimensions: {config.hidden_dim}\n"
        f"  Learning rate: {config.learning_rate}\n"
        f"  Discount factor: {config.discount_factor}\n"
        f"  Target update frequency: {config.target_update_freq} gradient updates\n\n"
        "Experience replay\n"
        f"  Buffer capacity: {config.replay_capacity:,} transitions\n"
        f"  Batch size: {config.replay_batch_size:,}\n"
        f"  Warm-up: {config.replay_batch_size:,} transitions (batch size)\n"
        "  Training frequency: every environment step after warm-up\n\n"
        "Exploration\n"
        f"  Current epsilon: {state.epsilon:.5f}\n"
        f"  Minimum epsilon: {config.min_epsilon}\n"
        f"  Epsilon decay: {config.epsilon_decay}\n"
        f"  Episodes to reach minimum epsilon: {epsilon_episodes:,}\n\n"
        "Logging & checkpoints\n"
        f"  Satisfactory reward threshold: {config.satisfactory_reward_threshold}\n"
        f"  Log interval: {config.log_interval}\n"
        f"  Save interval: {config.save_interval}\n"
        f"  Save path: {config.resolved_save_path(CURRENT_DIR)}"
    )


def main() -> None:
    """Load configuration, resume a compatible checkpoint, and train."""
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
            recent_rewards=deque(maxlen=config.log_interval),
        )

        if not args.new_model and save_path.exists():
            print(f"Checking existing checkpoint: {save_path}")
            loaded = load_checkpoint(save_path, config.device)
            meta_data = loaded.get("meta", {})
            if not isinstance(meta_data, dict):
                raise ValueError("Checkpoint metadata must be a dictionary.")

            saved_config = meta_data.get("config", {})
            if not isinstance(saved_config, dict):
                raise ValueError("Checkpoint configuration must be a dictionary.")

            can_resume, mismatched_fields = config.can_resume_with(saved_config)
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
            state.completed_episodes = int(meta_data.get("episodes", 0))
            state.epsilon = float(
                meta_data.get("epsilon", config.initial_epsilon)
            )
            state.total_reward_sum = float(
                meta_data.get("total_reward_sum", 0.0)
            )
            state.total_successful_episodes = int(
                meta_data.get("total_successful_episodes", 0)
            )
            print(
                f"  Resuming from episode {state.completed_episodes:,}; "
                f"restored {len(agent.replay_buffer):,} replay transitions."
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
            print(f"\nTraining completed: {state.completed_episodes:,} episodes")
            print(f"Overall average reward: {average_reward:.2f}")
            print(
                f"Total satisfactory episodes "
                f"(>= {config.satisfactory_reward_threshold}): "
                f"{state.total_successful_episodes}"
            )
            print(f"Final epsilon: {state.epsilon:.5f}")
            print(f"Gradient updates: {agent.steps:,}")
            print(f"Replay transitions retained: {len(agent.replay_buffer):,}")
            print(f"Model checkpoint saved to: {save_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import load_config
from .env import GridNavEnv


def main():
    config = load_config()
    vec_size = max(1, min(config.vector_size, 10))

    if config.log_output:
        print(f"Initializing training with vector_size: {vec_size}")

    if vec_size <= 1:
        env = DummyVecEnv([lambda: GridNavEnv(config=config)])
    else:
        env = make_vec_env(
            GridNavEnv, n_envs=vec_size, env_kwargs={"config": config}
        )

    model = PPO(
        "MlpPolicy", env, device="cpu",
        verbose=1 if config.log_output else 0,
    )

    if config.log_output:
        print("Starting PPO training...")
    started = time.time()
    model.learn(total_timesteps=50_000)

    if config.log_output:
        print(f"Training completed in {time.time() - started:.2f} seconds.")

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ppo_grid_nav"
    model.save(str(model_path))

    if config.log_output:
        print(f"Model saved successfully to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()

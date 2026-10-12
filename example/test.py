import time
from pathlib import Path

import pygame
from stable_baselines3 import PPO

from .config import load_config
from .env import GridNavEnv


def run_test():
    config = load_config()

    test_fps = config.test_fps
    log_output = config.log_output

    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "models" / "ppo_grid_nav.zip"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Please run train.py first!"
        )

    if log_output:
        print(f"Loading model from {model_path}...")
        print(
            f"Running test mode at {test_fps} FPS "
            "(configured in config.json)."
        )
        
    model = PPO.load(str(model_path), device="cpu")
    
    # Initialize environment with human rendering enabled.
    env = GridNavEnv(config=config, render_mode="human")
    obs, info = env.reset()

    if log_output:
        print("--- Running Test Mode ---")
        print("Close the window or press ESC to exit.\n")

    running = True
    episode_reward = 0.0
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                running = False

        # Predict action using the trained model.
        action, _states = model.predict(obs, deterministic=True)

        # Step through the environment.
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            if log_output:
                print(
                    f"Episode finished! Total Reward: "
                    f"{episode_reward:.2f}"
                )

            obs, info = env.reset()
            episode_reward = 0.0
            time.sleep(0.5)

        # Control test speed via the configured frame rate.
        clock.tick(test_fps)

    env.close()

if __name__ == "__main__":
    run_test()

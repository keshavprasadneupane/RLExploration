import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import GridConfig, load_config
from .game import GridGame


class GridNavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        config: GridConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config if config is not None else load_config()
        self.game = GridGame(self.config, render_mode=render_mode)

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(max(self.game.width, self.game.height) - 1),
            shape=(4,),
            dtype=np.float32,
        )
        self.render_mode = render_mode

		
    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)
        return self.game.reset(seed=seed), {}

    def step(self, action):
        action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        obs, terminated, player_old, player_new, target_old, target_new = (
            self.game.step(action)
        )

        if terminated:
            reward = 10.0
        else:
            dist_old = np.linalg.norm(player_old - target_old)
            dist_new = np.linalg.norm(player_new - target_new)
            reward = -0.1 + (0.05 if dist_new < dist_old else -0.05)

        return obs, reward, terminated, False, {}

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()

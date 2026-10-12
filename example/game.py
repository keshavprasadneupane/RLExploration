import numpy as np
import pygame

from .config import GridConfig, load_config


class GridGame:
    def __init__(
        self,
        config: GridConfig | None = None,
        render_mode: str | None = None,
    ):
        self.config = config if config is not None else load_config()
        self.width = self.config.grid_width
        self.height = self.config.grid_height
        self.include_diagonal = self.config.include_diagonal
        self.log_output = self.config.log_output

        self.render_mode = render_mode
        self.window_size = 600
        self.cell_size = self.window_size // max(self.width, self.height)

        self.window = None
        self.clock = None
        self.rng = np.random.default_rng()
        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.target_pos = np.array([0, 0], dtype=np.int32)

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            pygame.display.set_caption("Grid Navigation Game")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        while True:
            self.player_pos = np.array([
                self.rng.integers(0, self.width),
                self.rng.integers(0, self.height),
            ], dtype=np.int32)
            self.target_pos = np.array([
                self.rng.integers(0, self.width),
                self.rng.integers(0, self.height),
            ], dtype=np.int32)
            if not np.array_equal(self.player_pos, self.target_pos):
                break

        return self.get_observation()

    def get_observation(self):
        return np.array([
            self.player_pos[0], self.player_pos[1],
            self.target_pos[0], self.target_pos[1],
        ], dtype=np.float32)

    def step(self, action):
        old_player_pos = self.player_pos.copy()
        old_target_pos = self.target_pos.copy()

        action_map = {
            0: (-1,  1), 1: ( 0,  1), 2: ( 1,  1),
            3: (-1,  0), 4: ( 0,  0), 5: ( 1,  0),
            6: (-1, -1), 7: ( 0, -1), 8: ( 1, -1),
        }
        dx, dy = action_map.get(action, (0, 0))

        if dx != 0 and dy != 0 and not self.include_diagonal:
            dx, dy = 0, 0

        new_x = np.clip(self.player_pos[0] + dx, 0, self.width - 1)
        new_y = np.clip(self.player_pos[1] + dy, 0, self.height - 1)
        self.player_pos = np.array([new_x, new_y], dtype=np.int32)

        terminated = np.array_equal(self.player_pos, self.target_pos)

        if self.render_mode == "human":
            self.render()

        return (
            self.get_observation(), terminated,
            old_player_pos, self.player_pos.copy(),
            old_target_pos, self.target_pos.copy(),
        )

    def render(self):
        if self.window is None:
            return

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for x in range(self.width + 1):
            pygame.draw.line(
                canvas, (200, 200, 200),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.height * self.cell_size), 1,
            )
        for y in range(self.height + 1):
            pygame.draw.line(
                canvas, (200, 200, 200),
                (0, y * self.cell_size),
                (self.width * self.cell_size, y * self.cell_size), 1,
            )

        pygame.draw.rect(
            canvas, (46, 204, 113),
            pygame.Rect(
                self.target_pos[0] * self.cell_size,
                self.target_pos[1] * self.cell_size,
                self.cell_size, self.cell_size,
            ),
        )
        pygame.draw.rect(
            canvas, (52, 152, 219),
            pygame.Rect(
                self.player_pos[0] * self.cell_size,
                self.player_pos[1] * self.cell_size,
                self.cell_size, self.cell_size,
            ),
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        if self.clock:
            self.clock.tick(30)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

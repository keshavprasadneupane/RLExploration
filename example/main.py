import pygame

from .config import load_config
from .game import GridGame


def main():
    config = load_config()
    game: GridGame = GridGame(config=config, render_mode="human")
    game.reset()

    if config.log_output:
        print("--- Standalone Pygame Mode ---")
        print("Controls:")
        print("  Numpad keys 1-9 or Number row 1-9:")
        print("    7  8  9  (Up-Left, Up, Up-Right)")
        print("    4  5  6  (Left, Stay, Right)")
        print("    1  2  3  (Down-Left, Down, Down-Right)")
        print("  Diagonals also supported via WASD combos (W+A, W+D, S+A, S+D)")
        print("Close window or press ESC to exit.\n")

    running = True
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_KP1): action = 0
                elif event.key in (pygame.K_2, pygame.K_KP2): action = 1
                elif event.key in (pygame.K_3, pygame.K_KP3): action = 2
                elif event.key in (pygame.K_4, pygame.K_KP4): action = 3
                elif event.key in (pygame.K_5, pygame.K_KP5): action = 4
                elif event.key in (pygame.K_6, pygame.K_KP6): action = 5
                elif event.key in (pygame.K_7, pygame.K_KP7): action = 6
                elif event.key in (pygame.K_8, pygame.K_KP8): action = 7
                elif event.key in (pygame.K_9, pygame.K_KP9): action = 8

        if action is None:
            keys = pygame.key.get_pressed()
            w, s = keys[pygame.K_w], keys[pygame.K_s]
            a, d = keys[pygame.K_a], keys[pygame.K_d]
            if w and a: action = 6
            elif w and d: action = 8
            elif s and a: action = 0
            elif s and d: action = 2
            elif w: action = 7
            elif s: action = 1
            elif a: action = 3
            elif d: action = 5

        if action is not None:
            _, terminated, p_old, p_new, t_old, t_new = game.step(action)
            if config.log_output:
                print(
                    f"Action: {action} | Player: {p_old} -> {p_new} | "
                    f"Target: {t_old} -> {t_new} | Reached: {terminated}"
                )
            if terminated:
                if config.log_output:
                    print("🎉 Target reached! Resetting game...")
                game.reset()

        game.render()

    game.close()


if __name__ == "__main__":
    main()

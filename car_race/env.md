# CarRacing

This environment is part of the Box2D environments. You can view the official documentation [here](https://gymnasium.farama.org/v0.28.0/environments/box2d/car_racing/).

## Overview

* **Action Space:** `Box([-1. 0. 0.], 1.0, (3,), float32)` (or `Discrete(5)`)
* **Observation Space:** `Box(0, 255, (96, 96, 3), uint8)`

```python
import gymnasium as gym
env = gym.make("CarRacing-v2")

```

---

## Description

The easiest control task to learn from pixels—a top-down racing environment. The generated track is randomized every episode.

Indicators at the bottom of the window display (from left to right): true speed, four ABS sensors, steering wheel position, and gyroscope.

*Tip: It’s a powerful rear-wheel-drive car—avoid pressing the accelerator and turning sharply at the same time.*

---

## Action Space

The action space can be continuous (default) or discrete:

### Continuous Action Space (`Box([-1. 0. 0.], 1.0, (3,), float32)`)

| Num | Action | Min | Max | Description |
| --- | --- | --- | --- | --- |
| **0** | Steering | -1.0 | 1.0 | `-1` is full left, `+1` is full right |
| **1** | Gas | 0.0 | 1.0 | Acceleration |
| **2** | Braking | 0.0 | 1.0 | Brake force |

### Discrete Action Space (`Discrete(5)`)

| Num | Action |
| --- | --- |
| **0** | Do nothing |
| **1** | Steer left |
| **2** | Steer right |
| **3** | Gas |
| **4** | Brake |

---

## Observation Space

The observation is a top-down `96x96` RGB image representing the car and the race track frame buffer.

---

## Rewards

* **Step Penalty:** `-0.1` reward every frame.
* **Track Tile Reward:** `+1000 / N` for every track tile visited (where `N` is the total number of tiles on the track).
* *Example:* Finishing a track in 732 frames yields a reward of $1000 - (0.1 \times 732) = 926.8$ points.

---

## Starting State & Episode Termination

* **Starting State:** The car starts at rest in the center of the road.
* **Episode Termination & Truncation:**
* The episode finishes when all track tiles are visited.
* Going far off the track into the playfield background results in a `-100` penalty and immediate termination/death.



---

## Arguments & Reset Options

* `lap_complete_percent`: Dictates the percentage of tiles that must be visited before a lap is considered complete.
* `domain_randomize=True`: Enables domain randomization, changing track and background colors on every reset.
* `continuous=False`: Converts the environment to use a discrete action space.

```python
import gymnasium as gym

# Initialize with domain randomization enabled
env = gym.make("CarRacing-v2", domain_randomize=True)

# Reset with custom options (requires domain_randomize=True on init)
env.reset(options={"randomize": True})   # Changes color scheme
env.reset(options={"randomize": False})  # Keeps current color scheme

```

---

## Version History & References

* **v1:** Changed track completion logic and added domain randomization (`0.24.0`).
* **v0:** Original release.
* **References:** Chris Campbell (2014), *Box2D Tutorial*. Created by Oleg Klimov.
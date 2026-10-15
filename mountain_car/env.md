# Mountain Car

This environment is part of the Classic Control environments. You can view the official documentation [here](https://www.google.com/search?q=https://gymnasium.farama.org/v0.28.0/environments/classic_control/mountain_car/).

## Overview

* **Action Space:** `Discrete(3)`
* **Observation Space:** `Box([-1.2 -0.07], [0.6 0.07], (2,), float32)`

```python
import gymnasium as gym
env = gym.make("MountainCar-v0")

```

---

## Description

The Mountain Car MDP is a deterministic control problem where a car is placed in a sinusoidal valley. Because the car's engine is not powerful enough to drive straight up the right hill, the agent must learn to **rock back and forth** to build up sufficient momentum to reach the goal flag on top of the right hill.

First appearing in Andrew Moore's 1990 PhD Thesis (*Efficient Memory-based Learning for Robot Control*), this classic task tests an algorithm's ability to handle sparse rewards.

---

## Action Space

The action space is **Discrete(3)**:

| Num | Action | Description |
| --- | --- | --- |
| **0** | Push Left | Accelerate to the left |
| **1** | No Acceleration | Don't apply force |
| **2** | Push Right | Accelerate to the right |

---

## Observation Space

The observation is a 2-dimensional vector corresponding to position and velocity:

| Num | Observation | Min | Max | Unit |
| --- | --- | --- | --- | --- |
| **0** | Car Position | -1.2 | 0.6 | meters ($m$) |
| **1** | Car Velocity | -0.07 | 0.07 | velocity ($v$) |

---

## Transition Dynamics

The environment physics follow these equations:

$$\text{velocity}_{t+1} = \text{velocity}_t + (\text{action} - 1) \times \text{force} - \cos(3 \times \text{position}_t) \times \text{gravity}$$

$$\text{position}_{t+1} = \text{position}_t + \text{velocity}_{t+1}$$

* **Constants:** `force = 0.001`, `gravity = 0.0025`.
* **Collisions:** Inelastic at boundaries—position is clipped to $[-1.2, 0.6]$, velocity is clipped to $[-0.07, 0.07]$, and velocity resets to `0` upon hitting either wall.

---

## Rewards & Episode End

* **Reward:** `-1` for each timestep until the goal is reached, incentivizing the agent to find the shortest path.
* **Starting State:** Position is assigned a uniform random value in $[-0.6, -0.4]$; starting velocity is always `0`.
* **Episode End Conditions:**
1. **Termination:** Car position reaches or exceeds `0.5` (the goal on top of the right hill).
2. **Truncation:** Episode length reaches `200` steps.



---

## Arguments & Version History

```python
import gymnasium as gym

env = gym.make('MountainCar-v0')

```

On reset, the `options` parameter allows the user to change bounds used to determine the new random state.

* **v0:** Initial version release (`1.0.0`).
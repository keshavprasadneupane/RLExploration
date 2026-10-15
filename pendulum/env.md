
# Pendulum

This environment is part of the Classic Control environments. You can view the official documentation [here](https://gymnasium.farama.org/v0.28.0/environments/classic_control/pendulum/).

## Overview

* **Action Space:** `Box(-2.0, 2.0, (1,), float32)`
* **Observation Space:** `Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)`

```python
import gymnasium as gym
env = gym.make("Pendulum-v1")
```

## Description

The inverted pendulum swingup problem is a classic problem in control theory. The system consists of a pendulum attached at one end to a fixed point, with the other end free. The pendulum starts in a random position, and the goal is to apply torque on the free end to swing it into an upright position, with its center of gravity directly above the fixed point.

The coordinate system used for the dynamics equations:
* $x, y$: Cartesian coordinates of the pendulum's end in meters.
* $\theta$: Angle in radians.
* $\tau$: Torque in $\text{N}\cdot\text{m}$ (positive counter-clockwise).

## Action Space

The action is an `ndarray` with shape `(1,)` representing the torque applied to the free end of the pendulum.

| Num | Action | Min | Max |
| :--- | :--- | :--- | :--- |
| 0 | Torque | -2.0 | 2.0 |

## Observation Space

The observation is an `ndarray` with shape `(3,)` representing the $x-y$ coordinates of the pendulum's free end and its angular velocity.

| Num | Observation | Min | Max |
| :--- | :--- | :--- | :--- |
| 0 | $x = \cos(\theta)$ | -1.0 | 1.0 |
| 1 | $y = \sin(\theta)$ | -1.0 | 1.0 |
| 2 | Angular Velocity | -8.0 | 8.0 |

## Rewards

The reward function is defined as:

$$
r = -(\theta^2 + 0.1 \cdot \dot{\theta}^2 + 0.001 \cdot \tau^2)
$$

where $\theta$ is the pendulum's angle normalized between $[-\pi, \pi]$ (with $0$ being the upright position).

* **Minimum reward:** $-(\pi^2 + 0.1 \cdot 8^2 + 0.001 \cdot 2^2) = -16.2736044$
* **Maximum reward:** $0$ (pendulum upright with zero velocity and no torque applied).

## Starting State & Truncation

* **Starting State:** A random angle in $[-\pi, \pi]$ and a random angular velocity in $[-1, 1]$.
* **Episode Truncation:** The episode truncates at $200$ time steps.

## Arguments

* `g`: Acceleration of gravity measured in $\text{m}\cdot\text{s}^{-2}$ used to calculate the pendulum dynamics. Default value is $g = 10.0$.

```python
import gymnasium as gym
env = gym.make('Pendulum-v1', g=9.81)
```

On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

## Version History

* **v1:** Simplified math equations, no difference in behavior.
* **v0:** Initial version release ($1.0.0$).

# CartPole

This environment is part of the Classic Control environments. You can view the official documentation [here](https://gymnasium.farama.org/v0.28.0/environments/classic_control/cart_pole/).

## Overview

* **Action Space:** `Discrete(2)`
* **Observation Space:** `Box([-4.8 -3.4e+38 -0.41 -3.4e+38], [4.8 3.4e+38 0.41 3.4e+38], (4,), float32)`

```python
import gymnasium as gym
env = gym.make("CartPole-v1")

```

## Description

This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in “Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems”. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

## Action Space

The action is an `ndarray` with shape `(1,)` taking values $0$ or $1$, indicating the direction of the fixed force the cart is pushed with.

| Num | Action | Description |
| --- | --- | --- |
| 0 | Push Left | Push cart to the left |
| 1 | Push Right | Push cart to the right |

*Note: The velocity that is reduced or increased by the applied force depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it.*

## Observation Space

The observation is an `ndarray` with shape `(4,)` corresponding to the following positions and velocities:

| Num | Observation | Min | Max |
| --- | --- | --- | --- |
| 0 | Cart Position | -4.8 | 4.8 |
| 1 | Cart Velocity | -Inf | Inf |
| 2 | Pole Angle | ~-0.418 rad (-24°) | ~0.418 rad (24°) |
| 3 | Pole Angular Velocity | -Inf | Inf |

*Note: While the ranges above denote the possible values for the observation space, the episode terminates if limits are breached:*

* *The cart x-position episode terminates if it leaves the $\pm 2.4$ range.*
* *The pole angle episode terminates if the pole angle is not in the range $\pm 0.2095$ rad ($\pm 12^\circ$).*

## Rewards

Since the goal is to keep the pole upright for as long as possible, a reward of **+1** is given for every step taken, including the termination step. The threshold for rewards is **475** for `v1`.

## Starting State & Episode End

* **Starting State:** All observations are assigned a uniformly random value in `(-0.05, 0.05)`.
* **Episode End Conditions:**
1. **Termination:** Pole Angle is greater than $\pm 12^\circ$.
2. **Termination:** Cart Position is greater than $\pm 2.4$ (center of the cart reaches the edge of the display).
3. **Truncation:** Episode length is greater than $500$ ($200$ for `v0`).



## Arguments

* Default environment creation:

```python
import gymnasium as gym
env = gym.make('CartPole-v1')

```

On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.
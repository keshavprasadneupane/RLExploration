# CartPole Q-Table — Command Reference

Run all commands from the repository root (`RLExploration/`).

## Default Configuration

### Training (`train.py`)

| Setting | Default |
|---|---:|
| Episodes | `50,000,000` |
| Learning rate | `0.1` |
| Discount factor | `0.99` |
| Initial epsilon | `1.0` |
| Minimum epsilon | `0.05` |
| Epsilon decay | `0.9999` |
| Early-exit reward | `1000.0` |
| Early-exit count | `100` |
| Log interval | `10,000` |
| Save path | `models/cartpole_q_table.npy` |

### Evaluation (`test.py`)

| Setting | Default |
|---|---:|
| Q-table path | `models/cartpole_q_table.npy` |
| Episodes | `100` |
| Render mode | `none` |

## 1. Training

Train the agent using the default configuration:

```bash
uv run python -m cart_pole.cart_pole_q_table.train
```

Display available training arguments:

```bash
uv run python -m cart_pole.cart_pole_q_table.train --help
```

Override the number of training episodes:

```bash
uv run python -m cart_pole.cart_pole_q_table.train --episodes 100000
```

## 2. Evaluation

Evaluate the trained agent using the default configuration:

```bash
uv run python -m cart_pole.cart_pole_q_table.test
```

Display available evaluation arguments:

```bash
uv run python -m cart_pole.cart_pole_q_table.test --help
```

Evaluate for 1,000 episodes:

```bash
uv run python -m cart_pole.cart_pole_q_table.test --episodes 1000
```

Render the agent continuously in a visible Gymnasium environment. Close the window or press `Ctrl+C` to stop:

```bash
uv run python -m cart_pole.cart_pole_q_table.test --render-mode human
```

Load a Q-table from a custom path:
and the path is relative to current working directory.

```bash
uv run python -m cart_pole.cart_pole_q_table.test \
    --q-table-path models/cartpole.npy
```

Render the agent using a custom Q-table:

```bash
uv run python -m cart_pole.cart_pole_q_table.test \
    --render-mode human \
    --q-table-path models/cartpole.npy
```

Relative paths supplied through `--q-table-path` are resolved from the current working directory.

## 3. Typical Workflow

```bash
# Train the agent
uv run python -m cart_pole.cart_pole_q_table.train

# Evaluate the trained agent
uv run python -m cart_pole.cart_pole_q_table.test

# Render the trained agent
uv run python -m cart_pole.cart_pole_q_table.test --render-mode human
```

Ensure the training script's save path matches the evaluation script's default Q-table path.

## 4. Running as a Python Module

The `-m` option runs a module through Python's module system, using its package path:

```bash
uv run python -m cart_pole.cart_pole_q_table.train
uv run python -m cart_pole.cart_pole_q_table.test
```

This supports package-relative imports, such as:

```python
from ..agent import CartPoleQLearningAgent
```


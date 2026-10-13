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
uv run python -m cart_pole.cart_pole_q_table.train -h
```

Override the number of training episodes:

```bash
uv run python -m cart_pole.cart_pole_q_table.train --episodes 100000
```

Save the Q-table to a custom path:

```bash
uv run python -m cart_pole.cart_pole_q_table.train --save-path models/cartpole_custom.npy
```

Refer to `-h` or `--help` for all available training options and their descriptions.

## 2. Evaluation

Evaluate the trained agent using the default configuration:

```bash
uv run python -m cart_pole.cart_pole_q_table.test
```

Display available evaluation arguments:

```bash
uv run python -m cart_pole.cart_pole_q_table.test -h
```

Evaluate for 1,000 episodes:

```bash
uv run python -m cart_pole.cart_pole_q_table.test --episodes 1000
```

Enable verbose output to print the reward for every evaluation episode:

```bash
uv run python -m cart_pole.cart_pole_q_table.test -v
```

The `-v` and `--verbose` flags are equivalent. Without verbose mode, individual episode rewards are not printed during non-rendered evaluation; summary statistics are displayed after evaluation.

Render the agent in a visible Gymnasium environment. Episode rewards are printed continuously. Close the window or press `Ctrl+C` to stop:

```bash
uv run python -m cart_pole.cart_pole_q_table.test --render-mode human
```

Load a Q-table from a custom path:

```bash
uv run python -m cart_pole.cart_pole_q_table.test --q-table-path models/cartpole_custom.npy
```

Combine options as needed, for example, to evaluate 1,000 episodes with verbose output using a custom Q-table:

```bash
uv run python -m cart_pole.cart_pole_q_table.test \
    --episodes 1000 \
    --verbose \
    --q-table-path models/cartpole_custom.npy
```

Refer to `-h` or `--help` for all available evaluation options and their descriptions.

**Path resolution:** Paths supplied through `--save-path` and `--q-table-path` are resolved relative to the corresponding script's directory when given as relative paths. Absolute paths are also supported, but relative paths are recommended.
# CartPole DQN — Command Reference

Run all commands from the repository root (`RLExploration/`).

## Default Configuration

### Training (`train.py`)

| Setting | Default |
|---|---:|
| Episodes | `50,000,000` |
| Hidden layer neurons | `8` |
| Target update frequency | `1,000` |
| Learning rate | `0.001` |
| Discount factor | `0.99` |
| Initial epsilon | `1.0` |
| Minimum epsilon | `0.05` |
| Epsilon decay | `0.9999` |
| Device | `cpu` |
| Early-exit reward | `500.0` |
| Early-exit count | `100` |
| Log interval | `10,000` |
| Save path | `models/cartpole_dqn.pth` |

### Evaluation (`test.py`)

| Setting | Default |
|---|---:|
| Model path | `models/cartpole_dqn.pth` |
| Episodes | `100` |
| Render mode | `none` |
| Device | `cpu` |
| Verbose output | Disabled |

## 1. Training

Train the agent using the default configuration:

```bash
uv run python -m cart_pole.train
```

Display available training arguments:

```bash
uv run python -m cart_pole.train -h
```

Override the number of training episodes:

```bash
uv run python -m cart_pole.train --episodes 100000
```

Save the model to a custom path:

```bash
uv run python -m cart_pole.train --save-path models/cartpole_custom.pth
```

Refer to `-h` or `--help` for all available training options and their descriptions.

## 2. Evaluation

Evaluate the trained agent using the default configuration:

```bash
uv run python -m cart_pole.test
```

Display available evaluation arguments:

```bash
uv run python -m cart_pole.test -h
```

Evaluate for 1,000 episodes:

```bash
uv run python -m cart_pole.test --episodes 1000
```

Render the agent in a visible Gymnasium environment. Close the window or press `Ctrl+C` to stop:

```bash
uv run python -m cart_pole.test --render-mode human
```

Load a model checkpoint from a custom path:

```bash
uv run python -m cart_pole.test --model-path models/cartpole_custom.pth
```

Refer to `-h` or `--help` for all available evaluation options and their descriptions.

## 3. Typical Workflow

```bash
# Train the agent
uv run python -m cart_pole.train

# Evaluate the trained agent
uv run python -m cart_pole.test

# Render the trained agent
uv run python -m cart_pole.test --render-mode human
```

Ensure the training script's save path matches the evaluation script's default model path.

## 4. Running as a Python Module

The `-m` option runs a module through Python's module system using its package path:

```bash
uv run python -m cart_pole.train
uv run python -m cart_pole.test
```

Run these commands from the repository root to ensure Python can resolve the package and its imports.


**Path resolution:** Paths supplied through `--save-path` and `--model-path` are resolved relative to the corresponding script's directory when given as relative paths. Absolute paths are also supported, but relative paths are recommended.
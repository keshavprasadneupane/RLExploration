# CartPole DQN — Command Reference

Run all commands from the repository root (`RLExploration/`).

## Configuration Management

Training hyperparameters are managed via a JSON configuration file (defaulting to `config.json` inside `cart_pole/cart_pole_dqn/`).

* **First-time setup:** Running the training script for the first time without an existing `config.json` will automatically generate a default configuration file, print a notification, and exit safely so you can customize your settings.
* **Resuming vs. New Model:** By default (`--new-model false`), the script checks the model checkpoint and compares its saved configuration with your `config.json`. If they match, training resumes from where it off. If a mismatch is found, or if you explicitly pass `--new-model`, it will notify you and create a fresh model.

### Default Configuration (`config.json`)

| Setting | Default | Description |
| --- | --- | --- |
| `episodes` | `50,000,000` | Maximum number of training episodes |
| `hidden_dim` | `8` | Number of neurons in the hidden layer |
| `target_update_freq` | `1,000` | Synchronize target network every N updates |
| `learning_rate` | `0.001` | Optimizer learning rate |
| `discount_factor` | `0.99` | Discount factor for future rewards |
| `initial_epsilon` | `1.0` | Initial exploration probability |
| `min_epsilon` | `0.1` | Minimum exploration probability |
| `epsilon_decay` | `0.9999` | Epsilon decay multiplier |
| `device` | `cpu` | PyTorch device (`cpu`, `cuda`, or `mps`) |
| `early_exit_reward` | `500.0` | Reward threshold for counting a successful episode |
| `early_exit_count` | `100` | Successful episodes required for early exit |
| `log_interval` | `10,000` | Print training statistics every N episodes |
| `save_path` | `models/cartpole_dqn.pth` | Path where the model checkpoint will be saved |

### Evaluation (`test.py`)

| Setting | Default |
| --- | --- |
| Model path | `models/cartpole_dqn.pth` |
| Episodes | `100` |
| Render mode | `none` |
| Device | `cpu` |
| Verbose output | Disabled |

---

## 1. Training

Train the agent using the default configuration file (`config.json`):

```bash
uv run python -m cart_pole.cart_pole_dqn.train

```

Use a custom configuration file:

```bash
uv run python -m cart_pole.cart_pole_dqn.train --config custom_config.json

```

Force the creation of a new model, bypassing resume checks:

```bash
uv run python -m cart_pole.cart_pole_dqn.train --new-model

```

Display available command-line arguments:

```bash
uv run python -m cart_pole.cart_pole_dqn.train -h

```

---

## 2. Evaluation

Evaluate the trained agent using the default configuration:

```bash
uv run python -m cart_pole.cart_pole_dqn.test

```

Display available evaluation arguments:

```bash
uv run python -m cart_pole.cart_pole_dqn.test -h

```

Evaluate for 1,000 episodes:

```bash
uv run python -m cart_pole.cart_pole_dqn.test --episodes 1000

```

Render the agent in a visible Gymnasium environment. Close the window or press `Ctrl+C` to stop:

```bash
uv run python -m cart_pole.cart_pole_dqn.test --render-mode human

```

Load a model checkpoint from a custom path:

```bash
uv run python -m cart_pole.cart_pole_dqn.test --model-path models/cartpole_custom.pth

```

Refer to `-h` or `--help` for all available evaluation options and their descriptions.

---

## 3. Typical Workflow

```bash
# 1. Generate default config.json (if it doesn't exist yet)
uv run python -m cart_pole.cart_pole_dqn.train

# (Optional) Edit config.json with your desired parameters, then run:
uv run python -m cart_pole.cart_pole_dqn.train

# Evaluate the trained agent
uv run python -m cart_pole.cart_pole_dqn.test

# Render the trained agent
uv run python -m cart_pole.cart_pole_dqn.test --render-mode human

```

Ensure the training script's save path in your configuration matches the evaluation script's default model path.

---

## 4. Running as a Python Module

The `-m` option runs a module through Python's module system using its package path:

```bash
uv run python -m cart_pole.cart_pole_dqn.train
uv run python -m cart_pole.cart_pole_dqn.test

```

Run these commands from the repository root to ensure Python can resolve the package and its imports.

**Path resolution:** Paths supplied through `save_path` in `config.json` and `--model-path` in evaluation are resolved relative to the corresponding script's directory when given as relative paths. Absolute paths are also supported, but relative paths are recommended.
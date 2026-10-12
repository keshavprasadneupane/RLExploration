# RLExploration

RLExploration is a hands-on reinforcement learning playground for experimenting with environments, algorithms, and intelligent agents. It is organized as a collection of independent projects within a single repository, allowing each experiment to evolve without tightly coupling it to the others.

Each project owns its implementation, configuration, and generated artifacts. Projects implement their own features and avoid relying on the internal functionality of other projects. Shared functionality is permitted when it provides a genuine benefit, but the default approach is to keep implementations isolated while maintaining consistent repository-wide conventions.

## Project Philosophy

RLExploration follows a few core principles:

- **Project independence:** Each project should be understandable, maintainable, and runnable without relying on another project's implementation.
- **Feature isolation:** Each project implements the functionality it needs instead of importing or reusing another project's internal implementation.
- **Minimal cross-project dependencies:** Avoid coupling projects through their internal modules. Shared utilities are acceptable when they provide genuinely reusable functionality without introducing unnecessary coupling.
- **Shared environment:** Projects use the repository's root-level Python environment and dependency configuration.
- **Flexible architecture:** Projects may use different algorithms, frameworks, configurations, directory structures, and training workflows according to their needs.
- **Consistent conventions:** Use predictable execution commands, explicit paths, structured configuration where appropriate, and project-local artifact storage.
- **Experimentation and learning:** Prioritize understandable implementations, reproducible experiments, and comparisons between reinforcement learning techniques.

## Repository Structure

Each project resides in its own directory at the repository root. Its internal structure depends on the complexity and requirements of the experiment.

For example:

```text
RLExploration/
├── example/
│   ├── __init__.py
│   ├── config.py
│   ├── config.json
│   ├── game.py
│   ├── env.py
│   ├── train.py
│   ├── test.py
│   └── models/
├── cart_pole/
│   └── ...
├── blackjack/
│   └── ...
├── pyproject.toml
├── uv.lock
└── README.md
```

This is an illustrative structure, not a mandatory template. Small projects may require only a few files, while more complex projects may have separate agents, environments, replay buffers, evaluation modules, tests, and checkpoints.

The root-level `pyproject.toml` and `uv.lock` define the repository's shared Python project and its resolved environment. Projects use this common environment while keeping their implementations and generated artifacts separate.

Directory separation does not automatically isolate implementation details. Each project should have clear boundaries and avoid importing another project's internal modules simply to reuse existing functionality.

## Running a Project

Run commands from the repository root using Python's module execution syntax:

```bash
uv run python -m example.main
uv run python -m example.train
uv run python -m example.test
```

Replace `example` with the package name of the project you want to run. For example, a project located in `blackjack/` with an appropriate package structure can be executed using:

```bash
uv run python -m blackjack.main
```

This approach uses the repository's Python environment managed by `uv` and allows Python to resolve package-relative imports correctly.

Run commands from the repository root unless a project explicitly documents otherwise.

## Importing Project Modules

Within a project, use relative imports to access modules belonging to the same package.

For example, inside `example/main.py`:

```python
from .config import load_config
from .game import GridGame
```

When importing a project's public functionality from an external script, use its package-qualified name:

```python
from example.config import GridConfig, load_config
from example.game import GridGame
from example.env import GridNavigationEnv
```

The directory must be importable as a Python package, typically by including an `__init__.py` file.

Although cross-project imports are technically possible, avoid them when they make one project dependent on another project's internal implementation. If multiple projects genuinely need the same functionality, consider extracting it into an explicitly maintained shared module with a clear interface.

Do not rely on the current working directory to locate source modules. Use the intended package structure and execution workflow.

## Configuration Paths

Configuration files should be resolved relative to the module or project that owns them, rather than relying on the process's current working directory. This allows a project to load its default configuration consistently when launched from the repository root or imported by another module.

For example, `example/config.py` can locate its default configuration file using `pathlib`:

```python
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class GridConfig:
    width: int
    height: int
    cell_size: int


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(
    config_path: str | Path | None = None,
) -> GridConfig:
    path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else CONFIG_PATH
    )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return GridConfig(**data)
```

Calling `load_config()` loads the default `config.json` located beside `config.py`. To use another configuration file, pass its path explicitly:

```python
# Load the default configuration.
config = load_config()

# Load a custom configuration using a relative path.
config = load_config("configs/experiment.json")

# Load a configuration using an absolute path.
config = load_config("/path/to/custom_config.json")
```

An explicitly supplied relative path is resolved against the current working directory. Use an absolute path when the configuration must be located independently of where the command is launched.

Parse the configuration once and pass the resulting configuration object to the components that need it:

```python
config = load_config("example/config.json")

game = GridGame(config)
env = GridNavigationEnv(config)
```

The constructor arguments above are illustrative. Use the actual constructors provided by the project.

Configuration formats are not restricted to JSON. Each project may use the format and configuration approach best suited to its requirements while keeping defaults and path resolution explicit.

## Model and Artifact Paths

Store generated models and experiment artifacts within the corresponding project rather than in a shared repository-level directory.

For example, a training script can define a project-local model directory:

```python
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "ppo_grid_nav"
model.save(str(model_path))
```

This stores the model under `example/models/`, keeping it separate from models generated by other projects.

Apply the same approach to logs, checkpoints, evaluation results, plots, and other generated files.

When loading a model, resolve its path consistently:

```python
model_path = PROJECT_DIR / "models" / "ppo_grid_nav"
model = PPO.load(str(model_path), env=env)
```

Use the appropriate model class and filename for the algorithm and framework.

Generated models and other large artifacts should be excluded from version control when appropriate. Keep source code, configuration, documentation, and selected reproducible results under version control.

## General Conventions

- **Project isolation:** Keep each project's implementation, configuration, and generated artifacts within its own directory.
- **Feature ownership:** Implement project-specific functionality within the project that uses it.
- **Shared environment:** Use the root-level Python environment and dependency configuration managed by `uv`.
- **Package imports:** Prefer relative imports within a project and package-qualified imports when accessing its public functionality.
- **Minimal coupling:** Avoid dependencies on another project's internal implementation.
- **Controlled sharing:** Share genuinely reusable functionality through explicit interfaces rather than importing arbitrary files between projects.
- **Module execution:** Prefer `uv run python -m package.module` from the repository root.
- **Path handling:** Use `Path(__file__).resolve().parent` for paths anchored to a module or project.
- **Configuration reuse:** Parse configuration into structured objects where appropriate and pass those objects to components instead of repeatedly loading the same file.
- **Artifact organization:** Store models, checkpoints, logs, and evaluation outputs with the project that generates them.
- **Reproducibility:** Record important hyperparameters, random seeds, and evaluation metrics where appropriate.
- **Flexibility:** Allow each project to use the architecture and tooling appropriate to its requirements instead of enforcing an unnecessarily rigid template.

The overall objective is to maintain a single repository for exploring reinforcement learning while keeping its projects independently understandable, easy to modify, and straightforward to extend.
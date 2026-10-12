from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GridConfig:
    """Settings for the Grid Navigation mini-project."""

    grid_width: int = 10
    grid_height: int = 10
    include_diagonal: bool = False
    log_output: bool = True
    vector_size: int = 1
    test_fps: int = 30

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GridConfig":
        """Create a config object from JSON data."""
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in values.items() if key in allowed}

        # Accept the older key spelling for backwards compatibility.
        if "include_diagonal" not in filtered and "IncludeDiagonal" in values:
            filtered["include_diagonal"] = values["IncludeDiagonal"]

        return cls(**filtered)


def load_config(config_path: str | Path | None = None) -> GridConfig:
    """Load this mini-project's JSON config into a GridConfig object."""
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).with_name("config.json")
    )

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in config file: {path}")

    return GridConfig.from_dict(data)

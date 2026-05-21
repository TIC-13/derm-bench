from pathlib import Path
from typing import Any, Dict, Tuple, Union

import yaml


DEFAULT_ML_CONFIG_PATH = Path("configuration/models_config.yaml")


def load_yaml_config(
    config_path: Union[str, Path] = DEFAULT_ML_CONFIG_PATH,
) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file. If not provided,
            the default machine learning configuration path is used.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML file is empty.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"YAML config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"YAML config is empty: {config_path}")

    return config


def load_model_config(
    model_name: str,
    config_path: Union[str, Path] = DEFAULT_ML_CONFIG_PATH,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load default parameters and search space for a specific model.

    Args:
        model_name: Name of the model inside the YAML ``models`` section.
        config_path: Path to the YAML configuration file.

    Returns:
        A tuple containing:
            - default_params: Dictionary with the model default parameters.
            - search_space: Dictionary with the Optuna search space definition.

    Raises:
        KeyError: If the YAML file does not contain a ``models`` section or if the requested model is not found.
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML file is empty.
    """
    config = load_yaml_config(config_path)

    if "models" not in config:
        raise KeyError("YAML must contain a 'models' section.")

    if model_name not in config["models"]:
        raise KeyError(f"Model '{model_name}' not found in YAML.")

    model_config = config["models"][model_name]

    default_params = dict(model_config.get("default_params", {}))
    search_space = dict(model_config.get("search_space", {}))

    return default_params, search_space


def suggest_from_yaml(
    trial,
    param_name: str,
    param_config: Dict[str, Any],
):
    """Create an Optuna suggestion from a YAML search-space entry.

    Args:
        trial: Optuna trial object.
        param_name: Name of the hyperparameter being suggested.
        param_config: Dictionary describing the search space for the parameter.

    Returns:
        The value suggested by Optuna for the given parameter.

    Raises:
        KeyError: If required fields such as ``type``, ``low``, ``high``, or ``choices`` are missing.
        ValueError: If the parameter type is not supported.
    """
    param_type = param_config["type"]

    if param_type == "int":
        kwargs = {}

        if "step" in param_config:
            kwargs["step"] = param_config["step"]

        return trial.suggest_int(
            param_name,
            param_config["low"],
            param_config["high"],
            **kwargs,
        )

    if param_type == "float":
        kwargs = {}

        if "log" in param_config:
            kwargs["log"] = param_config["log"]

        if "step" in param_config:
            kwargs["step"] = param_config["step"]

        return trial.suggest_float(
            param_name,
            param_config["low"],
            param_config["high"],
            **kwargs,
        )

    if param_type == "categorical":
        return trial.suggest_categorical(
            param_name,
            param_config["choices"],
        )

    raise ValueError(f"Unsupported search space type: {param_type}")
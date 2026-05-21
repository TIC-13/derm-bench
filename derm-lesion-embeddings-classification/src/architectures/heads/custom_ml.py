from pathlib import Path
from typing import Union

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.architectures.heads.ml_base_model import BaseModel
from src.architectures.heads.models_config_utils import (
    DEFAULT_ML_CONFIG_PATH,
    load_model_config,
    suggest_from_yaml,
)


class RandomForestModel(BaseModel):
    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_ML_CONFIG_PATH,
        **kwargs,
    ):
        self.default_params, self.search_space = load_model_config(
            model_name="random_forest",
            config_path=config_path,
        )

        self.default_params.update(kwargs)
        self.model = RandomForestClassifier(**self.default_params)

    def define_search_space(self, trial):
        return {
            name: suggest_from_yaml(trial, name, param_config)
            for name, param_config in self.search_space.items()
        }

    def build_model(self, params):
        merged = {**self.default_params, **params}
        return RandomForestClassifier(**merged)


class XGBoostModel(BaseModel):
    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_ML_CONFIG_PATH,
        **kwargs,
    ):
        self.default_params, self.search_space = load_model_config(
            model_name="xgboost",
            config_path=config_path,
        )

        self.default_params.update(kwargs)
        self.model = xgb.XGBClassifier(**self.default_params)

    def define_search_space(self, trial):
        return {
            name: suggest_from_yaml(trial, name, param_config)
            for name, param_config in self.search_space.items()
        }

    def build_model(self, params):
        merged = {**self.default_params, **params}
        return xgb.XGBClassifier(**merged)


class SVMModel(BaseModel):
    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_ML_CONFIG_PATH,
        **kwargs,
    ):
        self.default_params, self.search_space = load_model_config(
            model_name="svm",
            config_path=config_path,
        )

        self.default_params.update(kwargs)
        self.model = SVC(**self.default_params)

    def define_search_space(self, trial):
        return {
            name: suggest_from_yaml(trial, name, param_config)
            for name, param_config in self.search_space.items()
        }

    def build_model(self, params):
        merged = {**self.default_params, **params}
        return SVC(**merged)
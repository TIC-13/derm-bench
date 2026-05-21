from pathlib import Path
from typing import Union

import torch.nn as nn

from src.architectures.heads.models_config_utils import (
    DEFAULT_ML_CONFIG_PATH,
    load_model_config,
)


class MLPClassifier(nn.Module):
    """Simple fully-connected neural network for tabular embeddings."""

    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_ML_CONFIG_PATH,
        **kwargs,
    ):
        super().__init__()

        self.default_params, self.search_space = load_model_config(
            model_name="mlp",
            config_path=config_path,
        )

        self.default_params.update(kwargs)

        input_dim = self.default_params["input_dim"]
        hidden_dim1 = self.default_params["hidden_dim1"]
        hidden_dim2 = self.default_params["hidden_dim2"]
        output_dim = self.default_params["output_dim"]
        dropout = self.default_params["dropout"]

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x
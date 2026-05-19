import torch.nn as nn

class MLPClassifier(nn.Module):
    """Simple fully-connected neural network for tabular embeddings."""

    def __init__(
        self,
        input_dim: int = 6144,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        output_dim: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

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
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    random_state: int = 0


class RULLSTM(nn.Module):
    def __init__(self, n_features: int, config: LSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.head(last_step).squeeze(-1)


class LSTMBaseline:
    """Sequence model over a sliding window of cycles -> RUL at the window's end."""

    def __init__(self, n_features: int, config: LSTMConfig | None = None):
        self.config = config or LSTMConfig()
        torch.manual_seed(self.config.random_state)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = RULLSTM(n_features, self.config).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMBaseline":
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.config.batch_size, shuffle=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.config.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_t).cpu().numpy()

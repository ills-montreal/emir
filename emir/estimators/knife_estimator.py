import torch

from .knife import KNIFE
from collections import namedtuple

from typing import Tuple, List
from dataclasses import dataclass, field

@dataclass(frozen=True)
class KnifeArgs:
    batch_size: int = 16
    lr: float = 0.01
    device: str = "cpu"
    n_epochs: int = 100
    average: str = "var"
    cov_diagonal: str = "var"
    cov_off_diagonal: str = "var"
    optimize_mu: bool = False
    simu_params: List[str] = field(
        default_factory=lambda: [
            "source_data",
            "target_data",
            "method",
            "optimize_mu",
        ]
    )
    cond_modes: int = 8
    marg_modes: int = 8
    use_tanh: bool = True
    init_std: float = 0.01
    ff_residual_connection: bool = False
    ff_activation: str = "relu"
    ff_layer_norm: bool = True
    ff_layers: int = 2


class KNIFEEstimator:
    def __init__(self, args: KnifeArgs, x_dim: int, y_dim: int):
        """

        :param args:
        :param x_dim:
        :param y_dim:
        """

        self.args = args
        self.x_dim = x_dim
        self.y_dim = y_dim

    def eval(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
        """
        Mutual information between x and y

        :param x: torch.Tensor
        :param y: torch.Tensor
        :return: Tuple[float, float, float] mutual information, marginal entropy H(X), conditional entropy H(X|Y)
        """

        # Create model for MI estimation
        self.knife = KNIFE(self.args, self.x_dim, self.y_dim).to(self.args.device)

        # Fit the model
        loss = self.fit_estimator(x, y)

        # Move model back to CPU
        self.knife = self.knife.to("cpu")
        x, y = x.to("cpu"), y.to("cpu")

        with torch.no_grad():
            mutual_information, marg_ent, cond_ent = self.knife(x, y)

        return mutual_information.item(), marg_ent.item(), cond_ent.item()

    def fit_estimator(self, x, y) -> List[float]:
        """
        Fit the estimator to the data
        """

        optimizer = torch.optim.SGD(self.knife.parameters(), lr=self.args.lr)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=self.args.batch_size
        )

        losses = []

        for epoch in range(self.args.n_epochs):
            for x_batch, y_batch in train_loader:
                # move data to device
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)

                optimizer.zero_grad()
                loss = self.knife.learning_loss(x_batch, y_batch)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

        return losses



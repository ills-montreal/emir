import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import torch
from tqdm import tqdm

from .knife import KNIFE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass(frozen=True)
class KNIFEArgs:
    batch_size: int = 16
    lr: float = 0.01
    device: str = "cpu"

    stopping_criterion: str = "max_epochs"  # "max_epochs" or "early_stopping"
    n_epochs: int = 100
    eps: float = 1e-6

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
    def __init__(self, args: KNIFEArgs, x_dim: int, y_dim: int):
        """

        :param args:
        :param x_dim:
        :param y_dim:
        """

        self.args = args
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.recorded_loss: List[float] = []

    def eval(
        self, x: torch.Tensor, y: torch.Tensor, record_loss: Optional[bool] = False
    ) -> Tuple[float, float, float]:
        """
        Mutual information between x and y

        :param x: torch.Tensor
        :param y: torch.Tensor
        :return: Tuple[float, float, float] mutual information, marginal entropy H(X), conditional entropy H(X|Y)
        """

        # Create model for MI estimation
        self.knife = KNIFE(self.args, self.x_dim, self.y_dim).to(self.args.device)

        # Fit the model
        self.fit_estimator(x, y, record_loss=record_loss)

        # Move model back to CPU
        self.knife = self.knife.to("cpu")
        x, y = x.to("cpu"), y.to("cpu")

        with torch.no_grad():
            mutual_information, marg_ent, cond_ent = self.knife(x, y)

        return mutual_information.item(), marg_ent.item(), cond_ent.item()

    def fit_estimator(self, x, y, record_loss: Optional[bool] = False) -> List[float]:
        """
        Fit the estimator to the data
        """

        optimizer = torch.optim.SGD(self.knife.parameters(), lr=self.args.lr)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=self.args.batch_size
        )

        losses = []

        for epoch in trange(self.args.n_epochs, position=1, desc="Knife training"):
            epoch_loss = []
            for x_batch, y_batch in train_loader:
                # move data to device
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)

                optimizer.zero_grad()
                loss = self.knife.learning_loss(x_batch, y_batch)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                epoch_loss.append(loss.item())

            self.recorded_loss.append(sum(epoch_loss) / len(epoch_loss))
            logger.info("Epoch %d: loss = %f", epoch, loss.item())

            if self.args.stopping_criterion == "early_stopping":
                if (
                    epoch > 0
                    and abs(self.recorded_loss[-1] - self.recorded_loss[-2])
                    < self.args.eps
                ):
                    logger.info("Reached early stopping criterion")
                    break

        return losses

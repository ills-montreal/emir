import torch

from .knife import KNIFE
from collections import namedtuple

from typing import Tuple, List


# Create a namedtuple for the arguments
KNIFEArgs = namedtuple(
    typename="KNIFEArgs",
    field_names=[
        "batch_size",
        "lr",
        "device",
        "n_epochs",
        "average",
        "cov_diagonal",
        "cov_off_diagonal",
        "optimize_mu",
        "simu_params",
        "cond_modes",
        "marg_modes",
        "use_tanh",
        "init_std",
        "ff_residual_connection",
        "ff_activation",
        "ff_layer_norm",
        "ff_layers",
    ],
    defaults=[
        16,  # batch_size
        0.01,  # lr
        "cpu",  # device
        100,  # n_epochs
        "var",  # average
        "var",  # cov_diagonal
        "var",  # cov_off_diagonal
        False,  # optimize_mu
        ["source_data", "target_data", "method", "optimize_mu"],  # simu_params
        8,  # cond_modes
        8,  # marg_modes
        True,  # use_tanh
        0.01,  # init_std
        False,  # ff_residual_connection
        "relu",  # ff_activation
        True,  # ff_layer_norm
        2,  # ff_layers
    ],
)


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
        self.fit_estimator(x, y)

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



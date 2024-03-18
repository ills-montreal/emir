import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Literal

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from .knife import KNIFE


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass(frozen=True)
class KNIFEArgs:
    batch_size: int = 16
    lr: float = 0.01
    device: str = "cpu"

    stopping_criterion: Literal[
        "max_epochs", "early_stopping"
    ] = "max_epochs"  # "max_epochs" or "early_stopping"
    n_epochs: int = 10
    n_epochs_marg: int = 10
    eps: float = 1e-3
    n_epochs_stop: int = 1
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
    ff_dim_hidden: Optional[int] = 0
    margin_lr: float = 1e-3


class KNIFEEstimator:
    def __init__(
        self,
        args: KNIFEArgs,
        x_dim: int,
        y_dim: int,
        precomputed_marg_kernel: Optional[float] = None,
    ):
        """

        :param args:
        :param x_dim:
        :param y_dim:
        """

        self.args = args
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.recorded_loss: List[float] = []
        self.recorded_marg_ent: List[float] = []
        self.recorded_cond_ent: List[float] = []
        self.early_stop_iter = 0
        self.precomputed_marg_kernel = precomputed_marg_kernel

    def eval(
        self,
        x: torch.torch.Tensor,
        y: torch.torch.Tensor,
        record_loss: Optional[bool] = False,
        fit_only_marginal: Optional[bool] = False,
    ) -> Tuple[float, float, float]:
        """
        Mutual information between x and y

        :param x: torch.Tensor
        :param y: torch.Tensor
        :return: Tuple[float, float, float] mutual information, marginal entropy H(X), conditional entropy H(X|Y)
        """

        # Create model for MI estimation
        if (y == 0).logical_or(y == 1).all():
            kernel_type = "discrete"
        else:
            kernel_type = "gaussian"
        self.kernel_type = kernel_type

        self.knife = KNIFE(
            self.args,
            self.x_dim,
            self.y_dim,
            kernel_type=kernel_type,
            precomputed_marg_kernel=self.precomputed_marg_kernel,
        ).to(self.args.device)

        # Fit the model
        self.fit_estimator(
            x, y, record_loss=record_loss, fit_only_marginal=fit_only_marginal
        )

        self.knife = self.knife.to("cpu")
        x, y = x.to("cpu"), y.to("cpu")

        with torch.no_grad():
            mutual_information, marg_ent, cond_ent = self.knife(x, y)

        return mutual_information.item(), marg_ent.item(), cond_ent.item()

    def eval_per_sample(
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
            mutual_information = self.knife.pmi(x, y)

        return mutual_information.squeeze().cpu().detach().numpy()

    def early_stopping(
        self,
        loss: List[float],
        marg_ent: float,
    ) -> bool:
        """
        Check if the early stopping criterion is reached
        """
        if (
            self.args.stopping_criterion == "early_stopping"
            and len(loss) > 1
            and abs((loss[-1] - loss[-2]) / (marg_ent + 1e-8)) < self.args.eps
        ):
            self.early_stop_iter += 1
            if self.early_stop_iter >= self.args.n_epochs_stop:
                self.early_stop_iter = 0
                return True
        else:
            self.early_stop_iter = 0
        return False

    def fit_estimator(
        self,
        x,
        y,
        record_loss: Optional[bool] = False,
        fit_only_marginal: Optional[bool] = False,
    ) -> List[float]:
        """
        Fit the estimator to the data
        """

        train_set = TensorDataset(x, y)
        train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.SGD(self.knife.parameters(), lr=self.args.margin_lr)

        if (
            self.precomputed_marg_kernel is None
        ):  # If a marginal kernel is not precomputed, we train it
            for epoch in trange(self.args.n_epochs_marg):
                epoch_loss, epoch_marg_ent, epoch_cond_ent = self.fit_marginal(
                    train_loader, optimizer
                )

                # Log 50 values for the loss
                step = max(len(epoch_loss) // 50, 1)
                for i in range(0, len(epoch_loss), step):
                    stop_idx = min(i + step, len(epoch_loss))
                    self.recorded_loss.append(
                        torch.tensor(epoch_loss[i:stop_idx]).mean()
                    )
                    self.recorded_marg_ent.append(
                        torch.tensor(epoch_marg_ent[i:stop_idx]).mean()
                    )
                    self.recorded_cond_ent.append(0)

        self.knife.freeze_marginal()

        if not fit_only_marginal:  # If we want to fit the full KNIFE estimator
            # First, we compute the marginal entropy on the dataset (used in the early stopping criterion)
            with torch.no_grad():
                marg_ent = []
                for x_batch, y_batch in train_loader:
                    y_batch = y_batch.to(self.args.device)
                    marg_ent.append(self.knife.kernel_marg(y_batch))
                marg_ent = torch.tensor(marg_ent).mean()

            # Then, we fit the conditional kernel
            optimizer.param_groups[0]["lr"] = self.args.lr
            for epoch in trange(self.args.n_epochs):
                epoch_loss, epoch_marg_ent, epoch_cond_ent = self.fit_conditional(
                    train_loader, optimizer
                )
                # Log 50 values for the loss
                step = max(len(epoch_loss) // 50, 1)

                for i in range(0, len(epoch_loss), step):
                    stop_idx = min(i + step, len(epoch_loss))
                    self.recorded_loss.append(
                        torch.tensor(epoch_loss[i:stop_idx]).mean()
                    )
                    self.recorded_marg_ent.append(0)
                    self.recorded_cond_ent.append(
                        torch.tensor(epoch_cond_ent[i:stop_idx]).mean()
                    )

                logger.info(
                    "Epoch %d: loss = %f",
                    epoch,
                    torch.tensor(epoch_loss[i:stop_idx]).mean(),
                )

                if self.early_stopping(self.recorded_loss, marg_ent):
                    logger.info("Reached early stopping criterion")
                    break

    def fit_marginal(
        self,
        train_loader,
        optimizer,
    ):
        epoch_loss = []
        epoch_marg_ent = []
        epoch_cond_ent = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.args.device), y_batch.to(
                self.args.device
            )
            optimizer.zero_grad()
            loss = self.knife.kernel_marg(y_batch)
            marg_ent = loss
            cond_ent = 0
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
            epoch_marg_ent.append(marg_ent)
            epoch_cond_ent.append(cond_ent)
        return epoch_loss, epoch_marg_ent, epoch_cond_ent

    def fit_conditional(
        self,
        train_loader,
        optimizer,
    ):
        epoch_loss = []
        epoch_marg_ent = []
        epoch_cond_ent = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.args.device), y_batch.to(
                self.args.device
            )
            optimizer.zero_grad()
            loss = self.knife.kernel_cond(x_batch, y_batch)
            cond_ent = loss
            marg_ent = 0
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
            epoch_marg_ent.append(marg_ent)
            epoch_cond_ent.append(cond_ent)
        return epoch_loss, epoch_marg_ent, epoch_cond_ent

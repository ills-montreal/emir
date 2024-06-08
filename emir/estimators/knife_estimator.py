import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Literal

import torch
from tqdm import tqdm, trange

from .knife import KNIFE

from .ftensordataloader import FastTensorDataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass(frozen=True)
class KNIFEArgs:
    batch_size: int = 64
    eval_batch_size: int = 1024
    lr: float = 1e-3
    margin_lr: float = 1e-3

    device: str = "cpu"

    stopping_criterion: Literal["max_epochs", "early_stopping"] = (
        "max_epochs"  # "max_epochs" or "early_stopping"
    )
    n_epochs: int = 10
    n_epochs_marg: int = 5
    eps: float = 1e-3
    n_epochs_stop: int = 1
    average: str = "var"
    cov_diagonal: str = "var"
    cov_off_diagonal: str = ""

    optimize_mu: bool = False
    cond_modes: int = 8
    marg_modes: int = 8
    use_tanh: bool = True
    init_std: float = 0.01
    ff_residual_connection: bool = False
    ff_activation: str = "relu"
    ff_layer_norm: bool = True
    ff_layers: int = 2
    ff_dim_hidden: Optional[int] = 0


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

        self.knife = None
        self.kernel_type = None

    def eval(
        self,
        x: torch.torch.Tensor,
        y: torch.torch.Tensor,
        fit_only_marginal: Optional[bool] = False,
    ) -> Tuple[float, float, float]:
        """
        Mutual information between x and y

        :param x: torch.Tensor
        :param y: torch.Tensor
        :return: Tuple[float, float, float] mutual information, marginal entropy H(X), conditional entropy H(X|Y)
        :param fit_only_marginal:  If True, only the marginal kernel is fitted (requires precomputed_marg_kernel)
        """

        # Some sanity checks
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

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

        self.knife = torch.compile(self.knife)

        self.fit_estimator(x, y, fit_only_marginal=fit_only_marginal)

        with torch.no_grad():
            mutual_information, marg_ent, cond_ent = self.batched_eval(x, y)

        return mutual_information.item(), marg_ent.item(), cond_ent.item()

    def batched_eval(
        self, x, y, per_samples=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mutual information between x and y

        :param x: torch.Tensor
        :param y: torch.Tensor
        :return: Tuple[float, float, float] mutual information, marginal entropy H(X), conditional entropy H(X|Y)
        :param per_samples: If True, return the mutual information per sample
        """

        eval_loader = FastTensorDataLoader(x, y, batch_size=self.args.batch_size)

        mi, h2, h2h1 = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in tqdm(eval_loader, desc="Evaluating", position=-1):
                with torch.no_grad():
                    x_batch = x_batch.to(self.args.device, non_blocking=True)
                    y_batch = y_batch.to(self.args.device, non_blocking=True)
                    mutual_information, marg_ent, cond_ent = self.knife.forward_samples(
                        x_batch.to(self.args.device), y_batch.to(self.args.device)
                    )
                    mi.append(mutual_information)
                    h2.append(marg_ent)
                    h2h1.append(cond_ent)

        # average mi
        if not per_samples:
            mi = torch.cat(mi).mean()
            h2 = torch.cat(h2).mean()
            h2h1 = torch.cat(h2h1).mean()
        else:
            mi = torch.cat(mi)
            h2 = torch.cat(h2)
            h2h1 = torch.cat(h2h1)

        return mi, h2, h2h1

    def eval_per_sample(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Mutual information between x and y. For back compatibility with the previous version.

        :param x: torch.Tensor
        :param y: torch.Tensor
        :return: Tuple[List[float], List[float], List[float]]: pmi, h2, h2h1
        """

        mi, h2, h2h1 = self.batched_eval(x, y, per_samples=True)
        mi, h2, h2h1 = mi.cpu().tolist(), h2.cpu().tolist(), h2h1.cpu().tolist()

        return mi, h2, h2h1

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
        fit_only_marginal: Optional[bool] = False,
    ) -> None:
        """
        Fit the estimator to the data
        """

        train_loader = FastTensorDataLoader(
            x,
            y,
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        if (
            self.precomputed_marg_kernel is None
        ):  # If a marginal kernel is not precomputed, we train it
            optimizer = torch.optim.Adam(
                self.knife.kernel_marg.parameters(), lr=self.args.margin_lr
            )
            for _ in trange(self.args.n_epochs_marg):
                epoch_loss, epoch_marg_ent, epoch_cond_ent = self.fit_marginal(
                    train_loader, optimizer
                )
                self.recorded_loss.append(sum(epoch_loss) / len(epoch_loss))
                self.recorded_marg_ent.append(sum(epoch_marg_ent) / len(epoch_marg_ent))
                # self.recorded_cond_ent.append(sum(epoch_cond_ent) / len(epoch_cond_ent))

        self.knife.freeze_marginal()
        if not fit_only_marginal:  # If we want to fit the full KNIFE estimator
            # First, we compute the marginal entropy on the dataset (used in the early stopping criterion)
            with torch.no_grad():
                marg_ent = []
                for x_batch, y_batch in train_loader:
                    # to device
                    y_batch = y_batch.to(self.args.device)
                    marg_ent.append(self.knife.kernel_marg(y_batch))
                marg_ent = torch.tensor(marg_ent).mean()

            # Then, we fit the conditional kernel
            optimizer = torch.optim.Adam(
                self.knife.kernel_cond.parameters(), lr=self.args.lr
            )
            for epoch in trange(self.args.n_epochs):
                epoch_loss, epoch_marg_ent, epoch_cond_ent = self.fit_conditional(
                    train_loader, optimizer
                )

                self.recorded_loss.append(sum(epoch_loss) / len(epoch_loss))
                # self.recorded_marg_ent.append(sum(epoch_marg_ent) / len(epoch_marg_ent))
                self.recorded_cond_ent.append(sum(epoch_cond_ent) / len(epoch_cond_ent))

                logger.info("Epoch %d: loss = %f", epoch, self.recorded_loss[-1])

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
        for _, y_batch in train_loader:
            y_batch = y_batch.to(self.args.device, non_blocking=True)
            optimizer.zero_grad()
            loss = self.knife.kernel_marg(y_batch)
            marg_ent = loss
            cond_ent = 0
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
            epoch_marg_ent.append(marg_ent)
            epoch_cond_ent.append(cond_ent)

        # make item
        epoch_loss = [x.item() for x in epoch_loss]
        epoch_marg_ent = [x.item() for x in epoch_marg_ent]
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
            x_batch = x_batch.to(self.args.device)
            y_batch = y_batch.to(self.args.device)
            optimizer.zero_grad()
            loss = self.knife.kernel_cond(x_batch, y_batch)
            cond_ent = loss
            marg_ent = 0
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
            epoch_marg_ent.append(marg_ent)
            epoch_cond_ent.append(cond_ent)

        # make item
        epoch_loss = [x.item() for x in epoch_loss]
        epoch_cond_ent = [x.item() for x in epoch_cond_ent]

        return epoch_loss, epoch_marg_ent, epoch_cond_ent

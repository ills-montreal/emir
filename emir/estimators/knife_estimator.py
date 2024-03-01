import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Literal

import torch
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
    async_lr: float = 0.1


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

        with torch.no_grad():
            mutual_information, marg_ent, cond_ent = self.knife(x, y)

        return mutual_information.item(), marg_ent.item(), cond_ent.item()

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

        train_loader = FastTensorDataLoader(
            x,
            y,
            batch_size=self.args.batch_size,
        )
        optimizer = torch.optim.SGD(self.knife.parameters(), lr=self.args.async_lr)

        if (
            self.precomputed_marg_kernel is None
        ):  # If a marginal kernel is not precomputed, we train it
            for epoch in trange(self.args.n_epochs_marg):
                epoch_loss, epoch_marg_ent, epoch_cond_ent = self.fit_marginal(
                    train_loader, optimizer
                )
                self.recorded_loss.append(sum(epoch_loss) / len(epoch_loss))
                self.recorded_marg_ent.append(sum(epoch_marg_ent) / len(epoch_marg_ent))
                self.recorded_cond_ent.append(sum(epoch_cond_ent) / len(epoch_cond_ent))

        self.knife.freeze_marginal()

        if not fit_only_marginal:  # If we want to fit the full KNIFE estimator
            # First, we compute the marginal entropy on the dataset (used in the early stopping criterion)
            with torch.no_grad():
                marg_ent = []
                for x_batch, y_batch in train_loader:
                    marg_ent.append(self.knife.kernel_marg(y_batch))
                marg_ent = torch.tensor(marg_ent).mean()

            # Then, we fit the conditional kernel
            optimizer.param_groups[0]["lr"] = self.args.lr
            for epoch in trange(self.args.n_epochs):
                epoch_loss, epoch_marg_ent, epoch_cond_ent = self.fit_conditional(
                    train_loader, optimizer
                )

                self.recorded_loss.append(sum(epoch_loss) / len(epoch_loss))
                self.recorded_marg_ent.append(sum(epoch_marg_ent) / len(epoch_marg_ent))
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
        for x_batch, y_batch in train_loader:
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


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    torch.TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a Fasttorch.TensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A Fasttorch.TensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

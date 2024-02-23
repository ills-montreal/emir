import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Literal

import torch
from tqdm import tqdm, trange


from .knife import KNIFE
from .distances import TanimotoDistance, calculate_kmeans_inertia


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
    n_epochs: int = 100
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
    mean_sep: float = 1e1
    delta_kernel: bool = False
    async_prop_training: float = 0.0
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

        :param x: torch.torch.Tensor
        :param y: torch.torch.Tensor
        :return: Tuple[float, float, float] mutual information, marginal entropy H(X), conditional entropy H(X|Y)
        """

        # Create model for MI estimation
        if (y == 0).logical_or(y == 1).all():
            kernel_type = "tanimoto"
            if self.args.delta_kernel:
                kernel_type = "tanimoto_delta"
        else:
            if self.args.delta_kernel:
                kernel_type = "gaussian_delta"
            else:
                kernel_type = "gaussian"
        self.kernel_type = kernel_type


        self.knife = KNIFE(
            self.args,
            self.x_dim,
            self.y_dim,
            kernel_type=kernel_type,
            reg_conf=self.args.mean_sep,
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

        optimizer = torch.optim.AdamW(self.knife.parameters(), lr=self.args.lr)

        train_loader = FastTensorDataLoader(
            x,
            y,
            batch_size=self.args.batch_size,
        )
        epochs_marg_async = int(self.args.n_epochs * self.args.async_prop_training)
        optimizer = torch.optim.SGD(self.knife.parameters(), lr=self.args.async_lr)

        if self.precomputed_marg_kernel is None:
            for epoch in range(epochs_marg_async):
                epoch_loss = []
                epoch_marg_ent = []
                epoch_cond_ent = []
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    loss, _ = self.knife.kernel_marg(y)
                    marg_ent = loss
                    cond_ent = loss - loss
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss)
                    epoch_marg_ent.append(marg_ent)
                    epoch_cond_ent.append(cond_ent)

                self.recorded_loss.append(sum(epoch_loss) / len(epoch_loss))
                self.recorded_marg_ent.append(sum(epoch_marg_ent) / len(epoch_marg_ent))
                self.recorded_cond_ent.append(sum(epoch_cond_ent) / len(epoch_cond_ent))

        self.knife.freeze_marginal()
        if not fit_only_marginal:
            optimizer.param_groups[0]["lr"] = self.args.lr
            for epoch in range(self.args.n_epochs - epochs_marg_async):
                epoch_loss = []
                epoch_marg_ent = []
                epoch_cond_ent = []
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    loss, marg_ent, cond_ent = self.knife.learning_loss(
                        x_batch, y_batch
                    )
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss)
                    epoch_marg_ent.append(marg_ent)
                    epoch_cond_ent.append(cond_ent)

                self.recorded_loss.append(sum(epoch_loss) / len(epoch_loss))
                self.recorded_marg_ent.append(sum(epoch_marg_ent) / len(epoch_marg_ent))
                self.recorded_cond_ent.append(sum(epoch_cond_ent) / len(epoch_cond_ent))

                logger.info("Epoch %d: loss = %f", epoch, self.recorded_loss[-1])

                if self.early_stopping(self.recorded_loss, marg_ent):
                    logger.info("Reached early stopping criterion")
                    break


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


def get_num_modes_via_kmeans(y, n_modes, distance=None):
    """
    Get the number of modes in the data using k-means clustering

    :param y: torch.Tensor
    :param n_modes: int
    :param distance: Optional[Callable]
    :return: Tuple[torch.Tensor, torch.Tensor]
    """
    from torch_kmeans import KMeans

    all_inertias = []
    for n_mode in n_modes:
        if distance is not None:
            kmeans = KMeans(n_clusters=n_mode, distance=distance)
        else:
            kmeans = KMeans(n_clusters=n_mode)
        results = kmeans(y.unsqueeze(0), verbose=0)
        if distance is not None:
            inertia = calculate_kmeans_inertia(
                y.unsqueeze(0), results.centers, results.labels, distance()
            )
        else:
            inertia = results.inertia
        all_inertias.append(inertia.item())
    return all_inertias

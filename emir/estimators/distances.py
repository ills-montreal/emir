import torch
from typing import Optional

from torch_kmeans.utils.distances import BaseDistance
from torch_kmeans.utils.utils import rm_kwargs



class TanimotoDistance(BaseDistance):

    def __init__(self, **kwargs):
        kwargs = rm_kwargs(kwargs, ["is_inverted"])
        super().__init__(is_inverted=False, **kwargs)
        assert not self.is_inverted

    def compute_mat(
            self, query_emb: torch.Tensor, ref_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the batched Tanimoto distance between
        each pair of the two collections of row vectors."""

        raise NotImplementedError

    def pairwise_distance(
            self,
            query_emb: torch.Tensor,
            ref_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the pairwise distance between
        vectors v1, v2 using the Tanimoto distance"""
        mult = (query_emb * ref_emb).sum(dim=-1)
        diff = ((query_emb - ref_emb)**2).sum(dim=-1)
        return 1- mult / (diff+mult + 1e-8)

    def pairwise_distance2(
            self,
            query_emb: torch.Tensor,
            ref_emb: torch.Tensor,
            temperature: float = 1000.0,
    ) -> torch.Tensor:
        """Computes the pairwise distance between
        vectors v1, v2 using the Tanimoto distance"""
        mask = torch.sigmoid(query_emb * ref_emb * temperature) * 2 - 1
        return 1 - (mask * (query_emb + ref_emb)).sum(dim=-1) / ((2 - mask) * (query_emb + ref_emb)).sum(dim=-1)


def calculate_kmeans_inertia(x, centers, labels, distance: BaseDistance):
    bs, n, d = x.size()
    m = centers.size(1)
    centers_x = centers.squeeze(0)[labels.squeeze(0)]
    print(distance)
    inertia = distance.pairwise_distance(x, centers_x).mean()
    return inertia
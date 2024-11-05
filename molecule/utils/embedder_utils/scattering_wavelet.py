import os
from typing import List

import datamol as dm
from scipy.spatial.distance import pdist
from rdkit import Chem
import numpy as np
import time
import torch

from kymatio.scattering3d.utils import generate_weighted_sum_of_gaussians
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D
from kymatio.torch import HarmonicScattering3D

from tqdm import trange

overlapping_precision = 1e-1
sigma = 2.0


def get_positions_charges_from_mols(mols_list:List[dm.Mol], i0, i1):
    """
    Get the positions and charges of the atoms from a file.

    Credits to the authors of the kymatio library for this function.

    :param path: str
    :return: Tuple[torch.Tensor, torch.Tensor]
    """
    mols = mols_list[i0 : min(i1, len(mols_list))]
    n_molecules = len(mols)

    max_atoms = 0
    for i, mol in enumerate(mols):
        for c in mol.GetConformers():
            max = len(c.GetPositions())
            if max > max_atoms:
                max_atoms = max

    pos = np.zeros((n_molecules, max_atoms, 3))
    full_charges = np.zeros((n_molecules, max_atoms))

    for i, mol in enumerate(mols):
        for j, c in enumerate(mol.GetConformers()):
            pos[i, : len(c.GetPositions()), :] = c.GetPositions()
            full_charges[i, : len(c.GetPositions())] = [
                a.GetAtomicNum() for a in mol.GetAtoms()
            ]
            break

    mask = full_charges <= 2
    valence_charges = full_charges * mask

    mask = np.logical_and(full_charges > 2, full_charges <= 10)
    valence_charges += (full_charges - 2) * mask

    mask = np.logical_and(full_charges > 10, full_charges <= 18)
    valence_charges += (full_charges - 10) * mask

    min_dist = np.inf

    for i in range(n_molecules):
        n_atoms = np.sum(full_charges[i] != 0)
        pos_i = pos[i, :n_atoms, :]
        min_dist = min(min_dist, pdist(pos_i).min())
        if min_dist == 0:
            print(i)
            print(pos_i)
            print(pdist(pos_i))
            print(pdist(pos_i).min())
            print(min_dist)
            time.sleep(1)

    delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
    pos = pos * delta / min_dist

    return pos, valence_charges, full_charges


def get_scatt_embeddings(pos, valence_charges, full_charges):
    n_molecules = pos.shape[0]
    M, N, O = 192, 128, 96

    grid = np.mgrid[-M // 2 : -M // 2 + M, -N // 2 : -N // 2 + N, -O // 2 : -O // 2 + O]
    grid = np.fft.ifftshift(grid)

    J = 2
    L = 3
    integral_powers = [0.5, 1.0, 2.0, 3.0]

    scattering = HarmonicScattering3D(
        J=J, shape=(M, N, O), L=L, sigma_0=sigma, integral_powers=integral_powers
    )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: {}".format(device))
    scattering.to(device)

    batch_size = 8
    n_batches = int(np.ceil(n_molecules / batch_size))

    order_0, orders_1_and_2 = [], []
    print(
        "Computing solid harmonic scattering coefficients of "
        "{} molecules from the QM7 database on {}".format(
            n_molecules, "GPU" if use_cuda else "CPU"
        )
    )
    print(
        "sigma: {}, L: {}, J: {}, integral powers: {}".format(
            sigma, L, J, integral_powers
        )
    )

    for i in trange(n_batches):
        # Extract the current batch.
        start = i * batch_size
        end = min(start + batch_size, n_molecules)

        pos_batch = pos[start:end]
        full_batch = full_charges[start:end]
        val_batch = valence_charges[start:end]

        # Calculate the density map for the nuclear charges and transfer
        # to PyTorch.
        full_density_batch = generate_weighted_sum_of_gaussians(
            grid, pos_batch, full_batch, sigma
        )
        full_density_batch = torch.from_numpy(full_density_batch)
        full_density_batch = full_density_batch.to(device).float()

        # Compute zeroth-order, first-order, and second-order scattering
        # coefficients of the nuclear charges.
        full_order_0 = TorchBackend3D.compute_integrals(
            full_density_batch, integral_powers
        )
        full_scattering = scattering(full_density_batch)

        # Compute the map for valence charges.
        val_density_batch = generate_weighted_sum_of_gaussians(
            grid, pos_batch, val_batch, sigma
        )
        val_density_batch = torch.from_numpy(val_density_batch)
        val_density_batch = val_density_batch.to(device).float()

        # Compute scattering coefficients for the valence charges.
        val_order_0 = TorchBackend3D.compute_integrals(
            val_density_batch, integral_powers
        )
        val_scattering = scattering(val_density_batch)

        # Take the difference between nuclear and valence charges, then
        # compute the corresponding scattering coefficients.
        core_density_batch = full_density_batch - val_density_batch

        core_order_0 = TorchBackend3D.compute_integrals(
            core_density_batch, integral_powers
        )
        core_scattering = scattering(core_density_batch)

        # Stack the nuclear, valence, and core coefficients into arrays
        # and append them to the output.
        batch_order_0 = torch.stack((full_order_0, val_order_0, core_order_0), dim=-1)
        batch_orders_1_and_2 = torch.stack(
            (full_scattering, val_scattering, core_scattering), dim=-1
        )

        order_0.append(batch_order_0)
        orders_1_and_2.append(batch_orders_1_and_2)
    order_0 = torch.cat(order_0, dim=0)
    orders_1_and_2 = torch.cat(orders_1_and_2, dim=0)

    order_0 = order_0.cpu().numpy()
    orders_1_and_2 = orders_1_and_2.cpu().numpy()
    order_0 = order_0.reshape((n_molecules, -1))
    orders_1_and_2 = orders_1_and_2.reshape((n_molecules, -1))
    scattering_coef = np.concatenate([order_0, orders_1_and_2], axis=1)

    return (scattering_coef - scattering_coef.mean(axis=0)) / (
        scattering_coef.std(axis=0) + 1e-8
    )


def get_scatt_from_path(path: str, i0, i1):
    mols = dm.read_sdf(path)
    pos, valence_charges, full_charges = get_positions_charges_from_mols(mols, i0, i1)
    return get_scatt_embeddings(pos, valence_charges, full_charges)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ClinTox")
    parser.add_argument("--i-start", type=int, default=0)
    parser.add_argument("--n-mols", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()
    path = f"data/{args.dataset}/preprocessed.sdf"
    print("Computing scattering wavelet...")
    print(f"Dataset: {args.dataset}")
    print(f"Start index: {args.i_start}")
    print(f"Number of molecules: {args.n_mols}")
    scatt = get_scatt_from_path(path, args.i_start, args.i_start + args.n_mols)
    save_path = os.path.join(args.out_dir, args.dataset)
    os.makedirs(save_path, exist_ok=True)
    filename = f"scattering_wavelet_{args.i_start}_{args.i_start + args.n_mols}.npy"
    np.save(
        os.path.join(save_path, filename),
        scatt,
    )
    print("Done.")

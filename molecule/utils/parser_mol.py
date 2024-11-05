import os
import argparse
from emir.estimators import KNIFEArgs


def add_eval_cli_args(parser: argparse.ArgumentParser):
    """
    Parser for the eval command line interface.
    Will collect :
        - The list of models to compare
        - The list of descriptors to compare
        - The dataset to use
        - Other parameters for the evaluation
    :param parser: argparse.ArgumentParser
    :return: argparse.ArgumentParser
    """
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
    )

    parser.add_argument("--n-runs", type=int, default=1)

    parser.add_argument(
        "--X",
        type=str,
        nargs="+",
        default=["GNN", "BERT", "GPT", "Denoising", "ThreeD", "MolR", "MoleOOD"],
        help="List of models to compare",
    )

    parser.add_argument(
        "--Y",
        type=str,
        nargs="+",
        default=["GNN", "BERT", "GPT", "Denoising", "ThreeD", "MolR", "MoleOOD"],
        help="List of descriptors to compare",
    )

    parser.add_argument(
        "--compute-both-mi",
        action="store_true",
        help="Compute both MI(x1, x2) and MI(x2, x1)",
    )
    parser.set_defaults(compute_both_mi=False)

    parser.add_argument(
        "--dataset",
        type=str,
        default="hERG",
        help="Dataset to use",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output file",
    )

    parser.add_argument("--fp-length", type=int, default=1024)
    parser.add_argument("--n-jobs", type=int, default=1)

    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--wandb", action="store_true")

    return parser


def add_knife_args(parser: argparse.ArgumentParser):
    """
    Add the arguments for the emir model. The parameters will be transmitted to the KMIFEArgs class.
    :param parser:
    :return:
    """
    parser.add_argument("--cond-modes", type=int, default=2)
    parser.add_argument("--marg-modes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--n-epochs-marg", type=int, default=100)
    parser.add_argument("--ff-layers", type=int, default=2)
    parser.add_argument("--cov-diagonal", type=str, default="var")
    parser.add_argument("--cov-off-diagonal", type=str, default="")
    parser.add_argument("--optimize-mu", type=str, default="true")
    parser.add_argument("--ff-residual-connection", type=str, default="false")
    parser.add_argument("--ff-hidden-dim", type=int, default=512)
    parser.add_argument("--use-tanh", type=str, default="true")
    parser.add_argument("--stopping-criterion", type=str, default="early_stopping")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--n-epochs-stop", type=int, default=5)
    parser.add_argument("--margin-lr", type=float, default=0.01)
    return parser


def generate_knife_config_from_args(args: argparse.Namespace) -> KNIFEArgs:
    """
    Generate the knife config from the argparse.Namespace object
    :param args: argparse.Namespace
    :return: KNIFEArgs
    """
    knife_config = KNIFEArgs(
        cond_modes=args.cond_modes,
        marg_modes=args.marg_modes,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        n_epochs=args.n_epochs,
        n_epochs_marg=args.n_epochs_marg,
        ff_layers=args.ff_layers,
        ff_dim_hidden=args.ff_hidden_dim,
        cov_diagonal=args.cov_diagonal,
        cov_off_diagonal=args.cov_off_diagonal,
        optimize_mu=args.optimize_mu == "true",
        ff_residual_connection=args.ff_residual_connection == "true",
        use_tanh=args.use_tanh == "true",
        stopping_criterion=args.stopping_criterion,
        n_epochs_stop=args.n_epochs_stop,
        eps=args.eps,
        margin_lr=args.margin_lr,
    )
    return knife_config


def add_FF_downstream_args(parser: argparse.ArgumentParser):
    """
    Add the arguments for the Feed Forward model. The parameters will be transmitted to the FFConfig class.
    :param parser:
    :return:
    """
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-rate", type=float, default=0.2)
    parser.add_argument("--norm", type=str, default="layer")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--test-batch-size", type=int, default=256)

    parser.add_argument("--hpo-whole-config", type=str, default=None)

    return parser


def add_downstream_args(parser: argparse.ArgumentParser):
    """
    Add the arguments for the downstream model. The parameters will be transmitted to the FFConfig class.
    :param parser:
    :return:
    """
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=["TOX", "ADME"])
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument(
        "--embedders",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="Embedders to use",
    )
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--plot-loss", action="store_true")
    parser.set_defaults(plot_loss=False)

    parser.add_argument("--config", type=str, default="downstream_config.yaml")
    parser.add_argument("--split-method", type=str, default="random")
    return parser

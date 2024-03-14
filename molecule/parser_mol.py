import argparse
from emir.estimators import KNIFEArgs


def add_eval_cli_args(parser: argparse.ArgumentParser):
    """
    Parser for the eval command line interface.
    Will collect :
        - The list of models to compare
        - The list of descriptors to compare
        - The dataset to use
    :param parser: argparse.ArgumentParser
    :return: argparse.ArgumentParser
    """
    parser.add_argument("--n-runs", type=int, default=1)

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "ContextPred",
            "GPT-GNN",
            "GraphMVP",
            "GROVER",
            # "EdgePred", # This model is especially bad and makes visualization hard
            "AttributeMask",
            "GraphLog",
            "GraphCL",
            "InfoGraph",
            "Not-trained",
            "MolBert",
            "ChemBertMLM-5M",
            "ChemBertMLM-10M",
            "ChemBertMLM-77M",
            "ChemBertMTR-5M",
            "ChemBertMTR-10M",
            "ChemBertMTR-77M",
            "ChemGPT-1.2B",
            "ChemGPT-19M",
            "ChemGPT-4.7M",
            "DenoisingPretrainingPQCMv4",
            "FRAD_QM9",
        ],
        help="List of models to compare",
    )

    parser.add_argument(
        "--compute-both-mi",
        action="store_true",
        help="Compute both MI(x1, x2) and MI(x2, x1)",
    )
    parser.set_defaults(compute_both_mi=True)

    parser.add_argument(
        "--descriptors",
        type=str,
        nargs="+",
        default=[
            "ContextPred",
            "GPT-GNN",
            "GraphMVP",
            "GROVER",
            # "EdgePred", # This model is especially bad and makes visualization hard
            "AttributeMask",
            "GraphLog",
            "GraphCL",
            "InfoGraph",
            "Not-trained",
            "MolBert",
            "ChemBertMLM-5M",
            "ChemBertMLM-10M",
            "ChemBertMLM-77M",
            "ChemBertMTR-5M",
            "ChemBertMTR-10M",
            "ChemBertMTR-77M",
            "ChemGPT-1.2B",
            "ChemGPT-19M",
            "ChemGPT-4.7M",
            "DenoisingPretrainingPQCMv4",
            "FRAD_QM9",
            "MolR_gat",
            "MolR_gcn",
            "MolR_tag",
            "MoleOOD_OGB_GIN"
            "MoleOOD_OGB_GCN",
            "MoleOOD_OGB_SAGE",
            "ThreeDInfoMax"
        ],
        help="List of descriptors to compare",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ClinTox",
        help="Dataset to use",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output file",
    )

    parser.add_argument("--fp-length", type=int, default=1024)
    parser.add_argument("--mds-dim", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def add_knife_args(parser: argparse.ArgumentParser):
    """
    Add the arguments for the emir model. The parameters will be transmitted to the KMIFEArgs class.
    :param parser:
    :return:
    """
    parser.add_argument("--cond-modes", type=int, default=6)
    parser.add_argument("--marg-modes", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-epochs-marg", type=int, default=10)
    parser.add_argument("--ff-layers", type=int, default=2)
    parser.add_argument("--cov-diagonal", type=str, default="var")
    parser.add_argument("--cov-off-diagonal", type=str, default="")
    parser.add_argument("--optimize-mu", type=str, default="true")
    parser.add_argument("--ff-residual-connection", type=str, default="false")
    parser.add_argument("--ff-hidden-dim", type=int, default=0)
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
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-rate", type=float, default=0.0)
    parser.add_argument("--norm", type=str, default="layer")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=100)

    return parser



def add_downstream_args(parser: argparse.ArgumentParser):
    """
    Add the arguments for the downstream model. The parameters will be transmitted to the FFConfig class.
    :param parser:
    :return:
    """
    parser.add_argument("--datasets", type=str, nargs="+", default=["TOX"])
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

    return parser

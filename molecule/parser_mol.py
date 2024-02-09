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
    parser.add_argument("--n-runs", type=int, default=10)

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "Not-trained",
            "AttributeMask",
            "ContextPred",
            #"EdgePred",
            "GPT-GNN",
            "InfoGraph",
            "GraphCL",
            "GraphLog",
            "GraphMVP",
            "GROVER",
            "InfoGraph",
            "ChemBertMLM-5M",
            #"ChemBertMLM-10M",
            #"ChemBertMLM-77M",
            "ChemBertMTR-5M",
            #"ChemBertMTR-10M",
            #"ChemBertMTR-77M",
            "MolBert",
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
            "ecfp",
            "estate",
            "cats",
            "electroshape",
            "estate",
            "fcfp",
            "gobbi",
            "mordred",
            "pmapper",
            "erg",
            "rdkit",
            "topological",
            "avalon",
            "maccs",
            "secfp",
            "scaffoldkeys",
            "usr",
            "usrcat",
        ],
        help="List of descriptors to compare",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="hERG_Karim",
        help="Dataset to use",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output file",
    )

    parser.add_argument("--fp-length", type=int, default=1024)

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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--ff-layers", type=int, default=1)
    parser.add_argument("--cov-diagonal", type=str, default="var")
    parser.add_argument("--cov-off-diagonal", type=str, default="")
    parser.add_argument("--optimize-mu", type=str, default="false")
    parser.add_argument("--ff-residual-connection", type=str, default="false")
    parser.add_argument("--use-tanh", type=str, default="true")
    parser.add_argument("--stopping-criterion", type=str, default="early_stopping")
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
        ff_layers=args.ff_layers,
        cov_diagonal=args.cov_diagonal,
        cov_off_diagonal=args.cov_off_diagonal,
        optimize_mu=args.optimize_mu,
        ff_residual_connection=args.ff_residual_connection == "true",
        use_tanh=args.use_tanh,
    )
    return knife_config

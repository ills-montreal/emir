import pandas as pd
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_paths", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output_path", type=Path, required=True, default="merged_df.csv"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    df_paths = args.df_paths
    print(df_paths)

    dfs = [pd.read_csv(df_path) for df_path in df_paths]
    merged_df = pd.concat(dfs)
    merged_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()

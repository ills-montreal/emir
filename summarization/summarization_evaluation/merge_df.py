
import pandas as pd
from sys import argv
from pathlib import Path




def main():
    # open argv directory
    dir_path = Path(argv[1])
    output_path = Path(argv[2])

    # list all csv files in the directory
    csv_files = list(dir_path.glob("*.csv"))

    # parse all csv files
    dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0)
        dfs.append(df)

    # merge all dataframes
    df = pd.concat(dfs)

    # save the merged dataframe
    df.to_csv(output_path, index=True)

if __name__ == "__main__":
    main()


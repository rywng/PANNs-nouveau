"""Filter csv entries by blacklist.txt

blacklist.txt is a file containing paths to audio that shouldn't be used.
"""

import argparse
from os import path

import polars as pl


def filter(csv_input: str, blacklist: str):
    df_in = pl.read_csv(csv_input)
    with open(blacklist, "r") as infile:
        blacklist_ls = list([i.strip() for i in infile])

    print(blacklist_ls)
    df_in = df_in.filter(~pl.col("path").str.contains_any(blacklist_ls))
    write_path = path.join(
        path.dirname(csv_input),
        path.basename(csv_input).replace(".csv", "_filtered.csv"),
    )
    print(f"Written to {write_path}")
    df_in.write_csv(write_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="filter_blacklist")
    parser.add_argument("csv_input")
    parser.add_argument("blacklist")
    args = parser.parse_args()

    filter(args.csv_input, args.blacklist)

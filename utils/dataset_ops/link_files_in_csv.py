import argparse
import os

import polars as pl


def link_files_in_csv(csv_path: str, to_dir: str, path_label: str = "path", label_filter = None):
    df = pl.read_csv(csv_path)
    if label_filter:
        df = df.filter(pl.col("label") == label_filter)
    res_rows = df.get_column(path_label)

    for file_path in set(res_rows):
        os.symlink(file_path, os.path.join(to_dir, os.path.basename(file_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Link files inside csv in to given output dir"
    )
    parser.add_argument("csv_path")
    parser.add_argument("out_dir")
    parser.add_argument("--path_label", default="path")
    parser.add_argument("--filter", default=None, type=int)
    args = parser.parse_args()

    link_files_in_csv(args.csv_path, args.out_dir, path_label=args.path_label, label_filter=args.filter)

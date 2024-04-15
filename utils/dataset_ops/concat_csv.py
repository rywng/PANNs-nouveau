import polars as pl
import argparse
import os


def concat_csv(
    csvs: list,
    no_test_csvs: list,
    test_train_ratio: float = 0,
    output_dir: str = ".",
    downsample_threshold: int = 0,
):
    train_res = None
    test_res = None

    for csv in csvs:
        train_df: pl.DataFrame = pl.read_csv(csv).sample(fraction=1.0)

        if downsample_threshold != 0 and downsample_threshold < train_df.shape[0]:
            train_df = train_df.sample(n=downsample_threshold)

        if no_test_csvs and csv not in no_test_csvs:
            test_size = int(train_df.shape[0] * test_train_ratio)
            test_df = train_df.head(test_size)
            train_df = train_df.tail(-1 * test_size)
            if test_res is None:
                test_res = test_df
            else:
                test_res = test_res.vstack(test_df)
            print(f"Test dataset generated for {csv}")

        if train_res is None:
            train_res = train_df
        else:
            train_res = train_res.vstack(train_df)
        print(f"Train dataset generated for {csv}")

    if train_res is not None:
        train_res.write_csv(os.path.join(output_dir, "train.csv"))
    if test_res is not None:
        test_res.write_csv(os.path.join(output_dir, "test.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Concatenates different csvs into one, and optionally split into train and test csvs."
    )
    parser.add_argument("--no_test_csvs", action="append")
    parser.add_argument("--csv_files_dir", required=True)
    parser.add_argument("--test_train_ratio", type=float)
    parser.add_argument("--downsample_threshold", type=int)
    parser.add_argument("--output_dir", default=".")

    args = parser.parse_args()
    concat_csv(
        [
            f"{os.path.join(args.csv_files_dir, filename)}"
            for filename in os.listdir(args.csv_files_dir)
        ],
        args.no_test_csvs,
        args.test_train_ratio,
        args.output_dir,
        args.downsample_threshold,
    )

import polars as pl

df1 = pl.read_csv("1.csv")
print(df1.describe())

df2 = pl.read_csv("2.csv")
print(df2.describe())

df3 = pl.read_csv("3.csv")
print(df3.describe())

df3 = df3.sample(fraction=1)

test_size = 1024

conc_df = df1.vstack(df2).vstack(df3.tail(-1 * test_size))
conc_df.write_csv("train.csv")

df3.head(test_size).write_csv("negative_test.csv")

import polars as pl
df1 = pl.read_csv("1.csv")
print(df1.describe())

df2 = pl.read_csv("2.csv")
print(df2.describe())

df3 = pl.read_csv("3.csv")
print(df3.describe())

conc_df = df1.vstack(df2).vstack(df3)
conc_df.write_csv("concat.csv")

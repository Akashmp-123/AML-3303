import pandas as pd
import polars as pl
import time

# Load a sample of the NYC taxi dataset
csv_file = "yellow_tripdata_2021-07.csv"

# ------------------ POLARS MULTITHREAD ANALYSIS ------------------
start = time.time()
df_pl_full = pl.read_csv(csv_file, n_rows=100000)

# Ensure required columns exist and are of correct type
required_columns = ["passenger_count", "total_amount", "tip_amount"]
for col in required_columns:
    if col not in df_pl_full.columns:
        raise ValueError(f"Column '{col}' not found in the CSV file.")

# Cast columns to appropriate types if needed
df_pl_full = df_pl_full.with_columns([
    pl.col("passenger_count").cast(pl.Int64),
    pl.col("total_amount").cast(pl.Float64),
    pl.col("tip_amount").cast(pl.Float64)
])

# Define passenger groups and ensure all are strings
df_pl_full = df_pl_full.with_columns(
    pl.when(pl.col("passenger_count") >= 4)
      .then(pl.lit("4+"))
      .otherwise(pl.col("passenger_count").cast(pl.Utf8))
      .alias("passenger_group")
)

# Group by passenger_group and compute metrics in parallel
metrics = (
    df_pl_full
    .group_by("passenger_group")
    .agg([
        pl.col("total_amount").sum().alias("revenue"),
        pl.col("tip_amount").mean().alias("avg_tip"),
        pl.col("tip_amount").max().alias("max_tip"),
        pl.count().alias("trip_count")
    ])
    .sort("passenger_group")
)

end = time.time()
print("Polars multithreaded metrics:\n", metrics.to_pandas())
print("Polars multithreaded execution time:", end - start, "seconds")

# # ------------------ PANDAS ------------------
# start = time.time()
# df_pd = pd.read_csv(csv_file, nrows=100000)  # Load first 100k rows
# # Example operation: filter and compute average fare
# result_pd = df_pd[df_pd['passenger_count'] > 2]['total_amount'].mean()
# end = time.time()
# print("Pandas result:", result_pd)
# print("Pandas execution time:", end - start, "seconds")

# # ------------------ POLARS ------------------
# start = time.time()
# df_pl = pl.read_csv(csv_file, n_rows=100000)
# # Example operation: filter and compute average fare
# result_pl = df_pl.filter(pl.col('passenger_count') > 2).select(pl.col('total_amount').mean())
# end = time.time()
# print("Polars result:", result_pl)
# print("Polars execution time:", end - start, "seconds")

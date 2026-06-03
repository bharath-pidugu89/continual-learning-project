import os
import pandas as pd


def generate_summary_table():

    methods = [
        "baseline",
        "replay",
        "ewc"
    ]

    benchmarks = [
        "basic",
        "permuted",
        "rotated"
    ]

    rows = []

    for benchmark in benchmarks:

        for method in methods:

            metrics_file = (
                f"results/{method}/"
                f"{benchmark}/metrics.csv"
            )

            metrics_df = pd.read_csv(
                metrics_file
            )

            rows.append({

                "Benchmark":
                    benchmark,

                "Method":
                    method,

                "ACC":
                    round(
                        metrics_df["ACC"][0],
                        2
                    ),

                "BWT":
                    round(
                        metrics_df["BWT"][0],
                        2
                    ),

                "FWT":
                    round(
                        metrics_df["FWT"][0],
                        2
                    )
            })

    summary_df = pd.DataFrame(rows)

    os.makedirs(
        "visualizations/tables",
        exist_ok=True
    )

    summary_df.to_csv(
        "visualizations/tables/summary_table.csv",
        index=False
    )


    acc_table = summary_df.pivot(
        index="Benchmark",
        columns="Method",
        values="ACC"
    )

    bwt_table = summary_df.pivot(
        index="Benchmark",
        columns="Method",
        values="BWT"
    )

    fwt_table = summary_df.pivot(
        index="Benchmark",
        columns="Method",
        values="FWT"
    )

    acc_table.to_csv(
        "visualizations/tables/acc_comparison_table.csv"
    )

    bwt_table.to_csv(
        "visualizations/tables/bwt_comparison_table.csv"
    )

    fwt_table.to_csv(
        "visualizations/tables/fwt_comparison_table.csv"
    )
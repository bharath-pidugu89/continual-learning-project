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

            metrics_path = (

                f"results/{method}/"
                f"{benchmark}/metrics.csv"
            )

            df = pd.read_csv(
                metrics_path
            )

            row = {

                "Benchmark":
                    benchmark,

                "Method":
                    method,

                "ACC":
                    df["ACC"][0],

                "BWT":
                    df["BWT"][0],

                "FWT":
                    df["FWT"][0]
            }

            rows.append(row)

    summary_df = pd.DataFrame(rows)

    summary_df.to_csv(

        "visualizations/"
        "summary_table.csv",

        index=False
    )

    print(
        "\nSummary table generated."
    )
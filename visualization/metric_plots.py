import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_comparison():

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

    metrics = [
        "ACC",
        "BWT",
        "FWT"
    ]

    for benchmark in benchmarks:

        metric_values = {
            metric: []
            for metric in metrics
        }

        for method in methods:

            metrics_path = (

                f"results/{method}/"
                f"{benchmark}/metrics.csv"
            )

            df = pd.read_csv(
                metrics_path
            )

            for metric in metrics:

                metric_values[
                    metric
                ].append(df[metric][0])

        # ==================================
        # PLOT
        # ==================================

        x = range(len(methods))

        width = 0.25

        plt.figure(figsize=(10, 6))

        for i, metric in enumerate(metrics):

            plt.bar(

                [p + width * i for p in x],

                metric_values[metric],

                width=width,

                label=metric
            )

        plt.xticks(

            [p + width for p in x],

            methods
        )

        plt.ylabel("Score")

        plt.title(
            f"{benchmark.upper()} "
            f"Metric Comparison"
        )

        plt.legend()

        os.makedirs(
            "visualizations/metrics",
            exist_ok=True
        )

        plt.savefig(

            f"visualizations/metrics/"
            f"{benchmark}_metrics.png"
        )

        plt.close()
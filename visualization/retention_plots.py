import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_retention_curves():

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

    for benchmark in benchmarks:

        plt.figure(figsize=(10, 6))

        for method in methods:

            matrix_path = (

                f"results/{method}/"
                f"{benchmark}/"
                f"accuracy_matrix.csv"
            )

            matrix = pd.read_csv(
                matrix_path
            ).values

            final_row = matrix[-1]

            tasks = list(
                range(1, len(final_row) + 1)
            )

            plt.plot(

                tasks,

                final_row,

                marker='o',

                label=method
            )

        plt.xlabel("Task ID")

        plt.ylabel("Final Accuracy")

        plt.title(
            f"{benchmark.upper()} "
            f"Retention Curves"
        )

        plt.legend()

        os.makedirs(
            "visualizations/retention",
            exist_ok=True
        )

        plt.savefig(

            f"visualizations/retention/"
            f"{benchmark}_retention.png"
        )

        plt.close()
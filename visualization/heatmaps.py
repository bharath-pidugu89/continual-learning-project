import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_heatmaps():

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

        for method in methods:

            matrix_path = (

                f"results/{method}/"
                f"{benchmark}/"
                f"accuracy_matrix.csv"
            )

            matrix = pd.read_csv(
                matrix_path
            )

            plt.figure(figsize=(8, 6))

            sns.heatmap(

                matrix,

                annot=True,

                fmt=".2f",

                cmap="viridis"
            )

            plt.title(

                f"{method.upper()} - "
                f"{benchmark.upper()}"
            )

            plt.xlabel("Evaluated Task")

            plt.ylabel("Trained Task")

            os.makedirs(
                "visualizations/heatmaps",
                exist_ok=True
            )

            plt.savefig(

                f"visualizations/heatmaps/"
                f"{method}_{benchmark}.png"
            )

            plt.close()
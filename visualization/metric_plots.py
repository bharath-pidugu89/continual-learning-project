import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {
    "baseline": "#E74C3C",
    "replay": "#3498DB",
    "ewc": "#2ECC71"
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 12
})


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

    output_dir = "visualizations/metrics"
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in benchmarks:

        df_all = []

        for method in methods:

            metrics_file = (
                f"results/{method}/"
                f"{benchmark}/metrics.csv"
            )

            df = pd.read_csv(metrics_file)

            df["Method"] = method

            df_all.append(df)

        metrics_df = pd.concat(df_all)

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        for idx, metric in enumerate(metrics):

            values = metrics_df[metric]

            bars = ax.bar(
                x + idx * width,
                values,
                width,
                label=metric
            )

            for bar in bars:

                height = bar.get_height()

                ax.annotate(
                    f"{height:.2f}",
                    xy=(
                        bar.get_x() +
                        bar.get_width() / 2,
                        height
                    ),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    fontsize=9
                )

        ax.set_title(
            f"{benchmark.upper()} Dataset\nACC / BWT / FWT Comparison",
            pad=20
        )

        ax.set_ylabel("Metric Value")

        ax.set_xticks(
            x + width
        )

        ax.set_xticklabels(
            [m.upper() for m in methods]
        )

        ax.grid(
            linestyle='--',
            alpha=0.4
        )

        ax.legend()

        plt.tight_layout()

        plt.savefig(
            f"{output_dir}/{benchmark}_metrics.png",
            dpi=300
        )

        plt.close()
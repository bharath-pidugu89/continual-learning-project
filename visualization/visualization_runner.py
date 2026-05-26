from visualization.metric_plots import (
    plot_metric_comparison
)

from visualization.retention_plots import (
    plot_retention_curves
)

from visualization.heatmaps import (
    generate_heatmaps
)

from visualization.summary_tables import (
    generate_summary_table
)


def generate_all_visualizations():

    print(
        "\nGenerating Metric Plots..."
    )

    plot_metric_comparison()

    print(
        "Generating Retention Curves..."
    )

    plot_retention_curves()

    print(
        "Generating Heatmaps..."
    )

    generate_heatmaps()

    print(
        "Generating Summary Table..."
    )

    generate_summary_table()

    print(
        "\nALL VISUALIZATIONS GENERATED."
    )
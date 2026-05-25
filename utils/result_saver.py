import os
import pandas as pd


def save_experiment_results(
        method_name,
        benchmark_name,
        accuracy_matrix,
        metrics):

    base_path = (
        f"results/"
        f"{method_name}/"
        f"{benchmark_name}"
    )

    os.makedirs(
        base_path,
        exist_ok=True
    )

    # ====================================
    # SAVE ACCURACY MATRIX
    # ====================================

    accuracy_df = pd.DataFrame(
        accuracy_matrix
    )

    accuracy_df.to_csv(

        f"{base_path}/"
        f"accuracy_matrix.csv",

        index=False
    )

    # ====================================
    # SAVE METRICS
    # ====================================

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(

        f"{base_path}/metrics.csv",

        index=False
    )

    print(
        f"\nResults saved to: "
        f"{base_path}"
    )
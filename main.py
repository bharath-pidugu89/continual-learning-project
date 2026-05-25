import torch
import pandas as pd

from models.cnn_model import SimpleCNN

from benchmarks.basic_benchmark import (
    build_basic_benchmark
)

from experiments.sequential_trainer import (
    run_sequential_experiment
)

from metrics.metrics_summary import (
    compute_all_metrics
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Build benchmark
    task_manager = build_basic_benchmark()

    # Create model
    model = SimpleCNN().to(device)

    # Run continual learning experiment
    accuracy_matrix = run_sequential_experiment(
        model,
        task_manager,
        device,
        epochs=3
    )

    print("\n===== ACCURACY MATRIX =====\n")

    print(accuracy_matrix)
    
    metrics = compute_all_metrics(
    accuracy_matrix
    )
    
    accuracy_df = pd.DataFrame(
    accuracy_matrix
    )

    accuracy_df.to_csv(
        "results/tables/accuracy_matrix.csv",
        index=False
    )

    print("\n===== CONTINUAL LEARNING METRICS =====\n")

    for metric_name, value in metrics.items():

        print(
            f"{metric_name}: {value:.4f}"
        )
        metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        "results/tables/metrics.csv",
        index=False
    )


if __name__ == "__main__":
    main()
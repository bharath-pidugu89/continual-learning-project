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

from benchmarks.permuted_mnist import (
    build_permuted_mnist
)

from benchmarks.rotated_mnist import (
    build_rotated_mnist
)

from benchmarks.split_cifar100 import (
    build_split_cifar100
)

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    benchmark_type = "permuted"
    
    # Build benchmark
    if benchmark_type == "basic":

        task_manager = build_basic_benchmark()

        model = SimpleCNN()

    elif benchmark_type == "permuted":

        task_manager = build_permuted_mnist()

        model = SimpleCNN()

    elif benchmark_type == "rotated":

        task_manager = build_rotated_mnist()

        model = SimpleCNN()

    elif benchmark_type == "cifar100":

        task_manager = build_split_cifar100()

        model = SimpleCNN(
            input_channels=3,
            num_classes=100
    )

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
import torch

from models.cnn_model import SimpleCNN

from benchmarks.basic_benchmark import (
    build_basic_benchmark
)

from experiments.sequential_trainer import (
    run_sequential_experiment
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


if __name__ == "__main__":
    main()
import torch
import pandas as pd

from torchvision import datasets
from utils.datasets import load_task_datasets
from utils.plotting import plot_results

from methods.baseline import run_baseline
from methods.replay import run_replay
from methods.ewc_training import run_ewc

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    (task_a_train,
    task_a_train_loader,
    task_a_test_loader,
    task_b_train,
    task_b_train_loader,
    task_b_test_loader) = load_task_datasets(
        dataset_a=datasets.MNIST,
        dataset_b=datasets.FashionMNIST,
        batch_size=32)

    # BASELINE
    baseline_task_a,baseline_task_b = run_baseline(
        task_a_train_loader,
        task_a_test_loader,
        task_b_train_loader,
        task_b_test_loader,
        device
    )

    # REPLAY
    replay_task_a,replay_task_b = run_replay(
        task_a_train,
        task_a_train_loader,
        task_a_test_loader,
        task_b_train,
        task_b_test_loader,
        device
    )

    # EWC
    ewc_task_a,ewc_task_b = run_ewc(
        task_a_train_loader,
        task_a_test_loader,
        task_b_train_loader,
        task_b_test_loader,
        device
    )

    # Final comparison table
    comparison = pd.DataFrame({
        "Method": [
            "Baseline",
            "Replay Buffer",
            "EWC"
        ],
        "task_a Retention": [
            baseline_task_a,
            replay_task_a,
            ewc_task_a
        ],
        "task_b Accuracy": [
            baseline_task_b,
            replay_task_b,
            ewc_task_b
        ]
    })

    print("\n===== FINAL COMPARISON =====\n")
    print(comparison)

    comparison.to_csv(
        "results/tables/final_comparison.csv",
        index=False
    )

    plot_results(
        comparison["Method"],
        comparison["task_a Retention"],
        "Continual Learning Comparison",
        "results/graphs/final_comparison.png"
    )


if __name__ == "__main__":
    main()
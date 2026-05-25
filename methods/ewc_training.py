import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from models.cnn_model import SimpleCNN
from models.ewc import EWC

from training.train import train_model
from training.evaluate import evaluate_model



def run_ewc(
        task_a_train_loader,
        task_a_test_loader,
        task_b_train_loader,
        task_b_test_loader,
        device,
        task_a_name="Task A",
        task_b_name="Task B"):

    print("\n===== EWC EXPERIMENT =====\n")

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Task A
    print(f"Training on {task_a_name}...")

    train_model(
        model,
        task_a_train_loader,
        criterion,
        optimizer,
        device,
        epochs=3
    )

    task_a_before = evaluate_model(
        model,
        task_a_test_loader,
        device
    )

    print(
        f"{task_a_name} BEFORE "
        f"{task_b_name}: {task_a_before:.2f}%"
    )

    # Create EWC object
    ewc = EWC(
        model,
        task_a_train_loader,
        device
    )

    # Train Task B with EWC
    print(f"\nTraining on {task_b_name} WITH EWC...")

    train_model(
        model,
        task_b_train_loader,
        criterion,
        optimizer,
        device,
        epochs=3,
        ewc=ewc,
        lambda_ewc=1000
    )

    task_b_accuracy = evaluate_model(
        model,
        task_b_test_loader,
        device
    )

    task_a_after = evaluate_model(
        model,
        task_a_test_loader,
        device
    )

    print(f"{task_a_name} WITH EWC: {task_b_accuracy:.2f}%")
    print(f"{task_b_name} AFTER EWC Training: {task_a_after:.2f}%")

    results = pd.DataFrame({
        "Metric": [
            "Task A Before Task B",
            "Task B Accuracy",
            "Task A After EWC"
        ],
        "Accuracy": [
            task_a_before,
            task_b_accuracy,
            task_a_after
        ]
    })

    results.to_csv(
        "results/tables/ewc_results.csv",
        index=False
    )

    return task_a_after, task_b_accuracy
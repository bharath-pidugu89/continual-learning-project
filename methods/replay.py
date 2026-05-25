import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random

from torch.utils.data import (
    Subset,
    ConcatDataset,
    DataLoader
)

from models.cnn_model import SimpleCNN
from training.train import train_model
from training.evaluate import evaluate_model


def run_replay(
        task_a_train_dataset,
        task_a_train_loader,
        task_a_test_loader,
        task_b_train_dataset,
        task_b_test_loader,
        device,
        task_a_name="Task A",
        task_b_name="Task B",
        batch_size=32,
        buffer_size=2000):

    print("\n===== REPLAY BUFFER EXPERIMENT =====\n")

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

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

    # Replay Buffer
    indices = random.sample(
        range(len(task_a_train_dataset)),
        buffer_size
    )

    memory_buffer = Subset(
        task_a_train_dataset,
        indices
    )

    combined_dataset = ConcatDataset([
        task_b_train_dataset,
        memory_buffer
    ])

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Train Task B + Replay
    print(
        f"\nTraining on {task_b_name} "
        f"WITH Replay Buffer..."
    )

    train_model(
        model,
        combined_loader,
        criterion,
        optimizer,
        device,
        epochs=3
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

    print(
        f"{task_b_name} Accuracy WITH Replay: "
        f"{task_b_accuracy:.2f}%"
    )

    print(
        f"{task_a_name} AFTER Replay: "
        f"{task_a_after:.2f}%"
    )

    return task_a_after, task_b_accuracy
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from training.train import train_model
from training.evaluate import evaluate_model

from metrics.accuracy_matrix import (
    AccuracyMatrix
)

from metrics.metrics_summary import (
    compute_all_metrics
)

from methods.replay_buffer import (
    ReplayBuffer
)


def run_replay_experiment(
        model,
        task_manager,
        device,
        epochs=3,
        batch_size=32,
        memory_size=2000):

    print("\n===== REPLAY EXPERIMENT =====\n")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    num_tasks = task_manager.num_tasks()

    accuracy_matrix = AccuracyMatrix(
        num_tasks
    )

    replay_buffer = ReplayBuffer(
        memory_size=memory_size
    )

    # Sequential training
    for train_task_id in range(num_tasks):

        print(
            f"\n===== TRAINING TASK "
            f"{train_task_id + 1} ====="
        )

        current_loader = (
            task_manager.train_loaders[
                train_task_id
            ]
        )

        current_dataset = (
            current_loader.dataset
        )

        # Combine current dataset + replay memory
        combined_dataset = (
            replay_buffer.get_combined_dataset(
                current_dataset
            )
        )

        combined_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Train
        train_model(
            model,
            combined_loader,
            criterion,
            optimizer,
            device,
            epochs=epochs
        )

        # Add current task to replay memory
        replay_buffer.add_dataset(
            current_dataset
        )

        # Evaluate ALL learned tasks
        for eval_task_id in range(
                train_task_id + 1):

            test_loader = (
                task_manager.test_loaders[
                    eval_task_id
                ]
            )

            accuracy = evaluate_model(
                model,
                test_loader,
                device
            )

            accuracy_matrix.update(
                train_task_id,
                eval_task_id,
                accuracy
            )

            print(
                f"Task {eval_task_id + 1} "
                f"Accuracy: {accuracy:.2f}%"
            )

    metrics = compute_all_metrics(
        accuracy_matrix.get_matrix()
    )

    return {
        "accuracy_matrix":
            accuracy_matrix.get_matrix(),

        "metrics":
            metrics
    }
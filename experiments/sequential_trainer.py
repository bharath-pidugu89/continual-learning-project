import torch.nn as nn
import torch.optim as optim

from training.train import train_model
from training.evaluate import evaluate_model

from metrics.accuracy_matrix import AccuracyMatrix


def run_sequential_experiment(
        model,
        task_manager,
        device,
        epochs=3):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    num_tasks = task_manager.num_tasks()

    accuracy_matrix = AccuracyMatrix(
        num_tasks
    )

    # Sequential Task Training
    for train_task_id in range(num_tasks):

        print(f'\n===== TRAINING TASK '
            f'{train_task_id + 1} =====\n'
        )

        train_loader = task_manager.train_loaders[
            train_task_id
        ]

        train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epochs=epochs
        )

        # Evaluate on ALL learned tasks
        for eval_task_id in range(train_task_id + 1):

            test_loader = task_manager.test_loaders[
                eval_task_id
            ]

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
                f'Task {eval_task_id + 1} '
                f'Accuracy: {accuracy:.2f}%'
            )

    return accuracy_matrix.get_matrix()
import torch
import torch.nn as nn
import torch.optim as optim

from training.evaluate import evaluate_model

from metrics.accuracy_matrix import (
    AccuracyMatrix
)

from metrics.metrics_summary import (
    compute_all_metrics
)

from methods.ewc_utils import EWC


def run_ewc_experiment(
        model,
        task_manager,
        device,
        epochs=3,
        lambda_ewc=1000):

    print("\n===== EWC EXPERIMENT =====\n")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    num_tasks = task_manager.num_tasks()

    accuracy_matrix = AccuracyMatrix(
        num_tasks
    )

    ewc_tasks = []

    # Sequential Tasks
    for train_task_id in range(num_tasks):

        print(
            f"\n===== TRAINING TASK "
            f"{train_task_id + 1} ====="
        )

        train_loader = (
            task_manager.train_loaders[
                train_task_id
            ]
        )

        # TRAIN LOOP
        model.train()

        for epoch in range(epochs):

            for inputs, labels in train_loader:

                inputs = inputs.to(device)

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(
                    outputs,
                    labels
                )

                # EWC penalties from old tasks
                if len(ewc_tasks) > 0:

                    ewc_loss = 0

                    for old_task in ewc_tasks:

                        ewc_loss += (
                            old_task.penalty(model)
                        )

                    loss += (
                        lambda_ewc * ewc_loss
                    )

                loss.backward()

                optimizer.step()

        # STORE EWC INFORMATION
        ewc_task = EWC(
            model,
            train_loader,
            device
        )

        ewc_tasks.append(ewc_task)

        # Evaluate all learned tasks
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
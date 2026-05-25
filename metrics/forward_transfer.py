import numpy as np


def compute_forward_transfer(
        accuracy_matrix,
        random_baseline=0):

    num_tasks = accuracy_matrix.shape[0]

    fwt_sum = 0

    for task_id in range(1, num_tasks):

        previous_accuracy = accuracy_matrix[
            task_id - 1,
            task_id
        ]

        fwt_sum += (
            previous_accuracy -
            random_baseline
        )

    fwt = fwt_sum / (num_tasks - 1)

    return fwt
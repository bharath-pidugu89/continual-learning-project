import numpy as np


def compute_backward_transfer(
        accuracy_matrix):

    num_tasks = accuracy_matrix.shape[0]

    bwt_sum = 0

    for task_id in range(num_tasks - 1):

        final_accuracy = accuracy_matrix[
            num_tasks - 1,
            task_id
        ]

        original_accuracy = accuracy_matrix[
            task_id,
            task_id
        ]

        bwt_sum += (
            final_accuracy -
            original_accuracy
        )

    bwt = bwt_sum / (num_tasks - 1)

    return bwt
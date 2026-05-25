import numpy as np


def compute_average_accuracy(
        accuracy_matrix):

    num_tasks = accuracy_matrix.shape[0]

    final_accuracies = accuracy_matrix[
        num_tasks - 1,
        :num_tasks
    ]

    acc = np.mean(final_accuracies)

    return acc
from metrics.accuracy import (
    compute_average_accuracy
)

from metrics.backward_transfer import (
    compute_backward_transfer
)

from metrics.forward_transfer import (
    compute_forward_transfer
)


def compute_all_metrics(
        accuracy_matrix):

    acc = compute_average_accuracy(
        accuracy_matrix
    )

    bwt = compute_backward_transfer(
        accuracy_matrix
    )

    fwt = compute_forward_transfer(
        accuracy_matrix
    )

    return {
        "ACC": acc,
        "BWT": bwt,
        "FWT": fwt
    }
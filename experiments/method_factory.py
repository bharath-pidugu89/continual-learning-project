from methods.sequential_trainer import (
    run_sequential_experiment
)

from methods.replay import (
    run_replay_experiment
)

from methods.ewc import (
    run_ewc_experiment
)

from metrics.metrics_summary import (
    compute_all_metrics
)


def run_method(
        method_name,
        model,
        task_manager,
        device,
        epochs=3):

    if method_name == "baseline":

        accuracy_matrix = (
            run_sequential_experiment(
                model,
                task_manager,
                device,
                epochs=epochs
            )
        )

        metrics = compute_all_metrics(
            accuracy_matrix
        )

        return {
            "accuracy_matrix":
                accuracy_matrix,

            "metrics":
                metrics
        }

    elif method_name == "replay":

        return run_replay_experiment(
            model,
            task_manager,
            device,
            epochs=epochs
        )

    elif method_name == "ewc":

        return run_ewc_experiment(
            model,
            task_manager,
            device,
            epochs=epochs
        )

    else:

        raise ValueError(
            f"Unknown method: "
            f"{method_name}"
        )
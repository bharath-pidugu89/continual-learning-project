import torch

from models.cnn_model import (
    SimpleCNN
)

from experiments.benchmark_factory import (
    build_benchmark
)

from experiments.method_factory import (
    run_method
)

from utils.result_saver import (
    save_experiment_results
)


def run_all_experiments():

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    print(f"\nUsing device: {device}")

    # ====================================
    # METHODS
    # ====================================

    methods = [

        "baseline",

        "replay",

        "ewc"
    ]

    # ====================================
    # BENCHMARKS
    # ====================================

    benchmarks = [

        "basic",

        "permuted",

        "rotated"
    ]

    # ====================================
    # RUN ALL COMBINATIONS
    # ====================================

    for benchmark_name in benchmarks:

        print(
            f"\n================================="
        )

        print(
            f"BENCHMARK: "
            f"{benchmark_name.upper()}"
        )

        print(
            f"=================================\n"
        )

        # Build benchmark
        task_manager = build_benchmark(
            benchmark_name
        )

        for method_name in methods:

            print(
                f"\n---------------------------------"
            )

            print(
                f"METHOD: "
                f"{method_name.upper()}"
            )

            print(
                f"---------------------------------\n"
            )

            # ====================================
            # CREATE MODEL
            # ====================================

            model = SimpleCNN().to(device)

            # ====================================
            # RUN METHOD
            # ====================================

            results = run_method(
                method_name,
                model,
                task_manager,
                device,
                epochs=3
            )

            # ====================================
            # SAVE RESULTS
            # ====================================

            save_experiment_results(

                method_name=method_name,

                benchmark_name=benchmark_name,

                accuracy_matrix=(
                    results[
                        "accuracy_matrix"
                    ]
                ),

                metrics=results["metrics"]
            )

    print(
        "\nALL EXPERIMENTS COMPLETED."
    )
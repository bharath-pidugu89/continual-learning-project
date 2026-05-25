from benchmarks.basic_benchmark import (
    build_basic_benchmark
)

from benchmarks.permuted_mnist import (
    build_permuted_mnist
)

from benchmarks.rotated_mnist import (
    build_rotated_mnist
)

from benchmarks.split_cifar100 import (
    build_split_cifar100
)


def build_benchmark(
        benchmark_name):

    if benchmark_name == "basic":

        return build_basic_benchmark()

    elif benchmark_name == "permuted":

        return build_permuted_mnist(
            num_tasks=5
        )

    elif benchmark_name == "rotated":

        return build_rotated_mnist(
            num_tasks=5
        )

    elif benchmark_name == "cifar100":

        return build_split_cifar100(
            num_tasks=10
        )

    else:

        raise ValueError(
            f"Unknown benchmark: "
            f"{benchmark_name}"
        )
from torchvision import datasets

from utils.datasets import create_task
from benchmarks.task_manager import TaskManager


def build_basic_benchmark(batch_size=32):

    task_manager = TaskManager()

    # TASK 1
    mnist_task = create_task(
        datasets.MNIST,
        batch_size=batch_size,
        task_name="MNIST"
    )

    task_manager.add_task(
        mnist_task["train_loader"],
        mnist_task["test_loader"],
        mnist_task["task_name"]
    )

    # TASK 2
    fashion_task = create_task(
        datasets.FashionMNIST,
        batch_size=batch_size,
        task_name="FashionMNIST"
    )

    task_manager.add_task(
        fashion_task["train_loader"],
        fashion_task["test_loader"],
        fashion_task["task_name"]
    )

    return task_manager
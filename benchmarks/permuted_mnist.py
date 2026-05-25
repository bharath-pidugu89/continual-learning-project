import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from benchmarks.task_manager import TaskManager


class PermutePixels:

    def __init__(self, permutation):

        self.permutation = permutation

    def __call__(self, tensor):

        flattened = tensor.view(-1)

        permuted = flattened[self.permutation]

        return permuted.view(1, 28, 28)
    
def build_permuted_mnist(
    num_tasks=5,
    batch_size=32):

    task_manager = TaskManager()

    for task_id in range(num_tasks):

        permutation = torch.randperm(28 * 28)

        transform = transforms.Compose([

            transforms.ToTensor(),

            PermutePixels(permutation),

            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        task_manager.add_task(
            train_loader,
            test_loader,
            f"PermutedMNIST-{task_id+1}"
        )

    return task_manager
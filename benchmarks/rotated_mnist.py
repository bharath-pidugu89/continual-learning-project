from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from benchmarks.task_manager import TaskManager


def build_rotated_mnist(
        num_tasks=5,
        angle_step=15,
        batch_size=32):

    task_manager = TaskManager()

    for task_id in range(num_tasks):

        angle = task_id * angle_step

        transform = transforms.Compose([

            transforms.RandomRotation(
                (angle, angle)
            ),

            transforms.ToTensor(),

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
            f"RotatedMNIST-{angle}deg"
        )

    return task_manager
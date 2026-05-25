from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from benchmarks.task_manager import TaskManager


def build_split_cifar100(
        num_tasks=10,
        classes_per_task=10,
        batch_size=32):

    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    task_manager = TaskManager()

    for task_id in range(num_tasks):

        start_class = task_id * classes_per_task

        end_class = start_class + classes_per_task

        train_indices = [
            i for i, (_, label)
            in enumerate(train_dataset)
            if start_class <= label < end_class
        ]

        test_indices = [
            i for i, (_, label)
            in enumerate(test_dataset)
            if start_class <= label < end_class
        ]

        train_subset = Subset(
            train_dataset,
            train_indices
        )

        test_subset = Subset(
            test_dataset,
            test_indices
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False
        )

        task_manager.add_task(
            train_loader,
            test_loader,
            f"CIFAR100-{start_class}-{end_class-1}"
        )

    return task_manager
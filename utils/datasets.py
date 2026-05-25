from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_task(
        dataset_class,
        batch_size=32,
        transform=None,
        task_name="Task"):

    if transform is None:

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    train_dataset = dataset_class(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = dataset_class(
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

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "task_name": task_name
    }
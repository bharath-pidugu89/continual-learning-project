from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_task_datasets(
        dataset_a,
        dataset_b,
        batch_size=32):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # TASK A
    task_a_train = dataset_a(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    task_a_test = dataset_a(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # TASK B
    task_b_train = dataset_b(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    task_b_test = dataset_b(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # DATALOADERS
    task_a_train_loader = DataLoader(
        task_a_train,
        batch_size=batch_size,
        shuffle=True
    )

    task_a_test_loader = DataLoader(
        task_a_test,
        batch_size=batch_size,
        shuffle=False
    )

    task_b_train_loader = DataLoader(
        task_b_train,
        batch_size=batch_size,
        shuffle=True
    )

    task_b_test_loader = DataLoader(
        task_b_test,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        task_a_train,
        task_a_train_loader,
        task_a_test_loader,
        task_b_train,
        task_b_train_loader,
        task_b_test_loader
    )
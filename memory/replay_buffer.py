import random

from torch.utils.data import (
    ConcatDataset,
    Subset
)


class ReplayBuffer:

    def __init__(self, memory_size=2000):

        self.memory_size = memory_size

        self.datasets = []

    def add_dataset(self, dataset):

        indices = random.sample(
            range(len(dataset)),
            min(self.memory_size, len(dataset))
        )

        subset = Subset(dataset, indices)

        self.datasets.append(subset)

    def get_combined_dataset(
            self,
            current_dataset):

        if len(self.datasets) == 0:

            return current_dataset

        replay_dataset = ConcatDataset(
            self.datasets
        )

        combined_dataset = ConcatDataset([
            current_dataset,
            replay_dataset
        ])

        return combined_dataset
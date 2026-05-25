class TaskManager:

    def __init__(self):

        self.train_loaders = []
        self.test_loaders = []
        self.task_names = []

    def add_task(
            self,
            train_loader,
            test_loader,
            task_name):

        self.train_loaders.append(train_loader)

        self.test_loaders.append(test_loader)

        self.task_names.append(task_name)

    def num_tasks(self):

        return len(self.train_loaders)
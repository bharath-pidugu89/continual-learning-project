import numpy as np


class AccuracyMatrix:

    def __init__(self, num_tasks):

        self.matrix = np.zeros(
            (num_tasks, num_tasks)
        )

    def update(
            self,
            train_task_id,
            eval_task_id,
            accuracy):

        self.matrix[
            train_task_id,
            eval_task_id
        ] = accuracy

    def get_matrix(self):

        return self.matrix
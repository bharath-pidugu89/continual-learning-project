import torch


class EWC:

    def __init__(
            self,
            model,
            dataloader,
            device):

        self.model = model

        self.device = device

        self.params = {

            n: p.clone().detach()

            for n, p
            in model.named_parameters()

            if p.requires_grad
        }

        self.fisher = self.compute_fisher(
            dataloader
        )

    def compute_fisher(
            self,
            dataloader):

        fisher = {

            n: torch.zeros_like(p)

            for n, p
            in self.model.named_parameters()

            if p.requires_grad
        }

        self.model.eval()

        for inputs, labels in dataloader:

            inputs = inputs.to(self.device)

            labels = labels.to(self.device)

            self.model.zero_grad()

            outputs = self.model(inputs)

            loss = torch.nn.functional.cross_entropy(
                outputs,
                labels
            )

            loss.backward()

            for n, p in self.model.named_parameters():

                if p.grad is not None:

                    fisher[n] += (
                        p.grad.data.clone() ** 2
                    )

        for n in fisher:

            fisher[n] /= len(dataloader)

        return fisher

    def penalty(self, model):

        loss = 0

        for n, p in model.named_parameters():

            loss += (
                self.fisher[n] *
                (p - self.params[n]) ** 2
            ).sum()

        return loss
import torch

class EWC:

    def __init__(self,
                 model,
                 dataloader,
                 device):

        self.model = model
        self.device = device

        self.params = {
            n: p for n, p in model.named_parameters()
            if p.requires_grad
        }

        self.means = {}

        for n, p in self.params.items():
            self.means[n] = p.clone().detach()

        self.fisher = self.compute_fisher(dataloader)

    def compute_fisher(self, dataloader):

        fisher = {
            n: torch.zeros_like(p).to(self.device)
            for n, p in self.params.items()
        }

        self.model.eval()

        for images, labels in dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()

            outputs = self.model(images)

            loss = torch.nn.functional.cross_entropy(
                outputs,
                labels
            )

            loss.backward()

            for n, p in self.model.named_parameters():

                if p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)

        for n in fisher:
            fisher[n] = fisher[n] / len(dataloader)

        return fisher

    def penalty(self, model):

        loss = 0

        for n, p in model.named_parameters():

            loss += (
                self.fisher[n] *
                (p - self.means[n]).pow(2)
            ).sum()

        return loss
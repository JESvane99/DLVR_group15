from torch import nn


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    def forward(self, logits: nn.Parameter, targets: nn.Parameter):
        return self.bce(logits, targets)

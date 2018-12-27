import torch.nn as nn
import torch


class SoftCrossEntropy(nn.Module):
    def __init__(self, weight=None, reduce=True, average_size=True, reduction='mean'):
        super(SoftCrossEntropy, self).__init__()
        self.reduce = reduce
        self.average_size = average_size
        self.reduction = reduction
        if weight is not None:
            self.weight = weight
        else:
            self.weight = 1.

    def forward(self, varInput, varTarget):
        losstensor = - self.weight * varTarget * varInput

        if self.reduce is False:
            return losstensor
        else:
            if self.average_size:
                if self.reduction == 'mean':
                    return torch.mean(losstensor)
                elif self.reduction == 'sum':
                    return torch.sum(losstensor)
            else:
                if self.reduction == 'mean':
                    return torch.mean(losstensor, 1).view(losstensor.size(0), -1)
                elif self.reduction == 'sum':
                    return torch.sum(losstensor, 1).view(losstensor.size(0), -1)

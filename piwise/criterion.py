import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)


if __name__ == "__main__":
    target = torch.empty(4,256,256, dtype =torch.long).random_(0,21)
    inputdata = torch.randn(4,21,256,256)
    loss = nn.NLLLoss()
    result = loss(inputdata, target)
    print(result)
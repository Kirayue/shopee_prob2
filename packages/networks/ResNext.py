import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNext(nn.Module):
    def __init__(self):
        super(ResNext, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 43)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    model = ResNext()
    print(model)
    print(model(torch.rand(8, 3, 224, 224)).shape)

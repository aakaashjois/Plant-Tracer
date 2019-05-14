import torch
from torchvision import models
from torch import nn


class GoTurnRemix(nn.Module):
    """
        Create a model based on GOTURN. The GOTURN architecture used a CaffeNet while GoTurnRemix uses AlexNet.
        The rest of the architecture is the similar to GOTURN. A PyTorch implementation of GOTURN can be found at:

        https://github.com/aakaashjois/PyTorch-GOTURN
    """

    def __init__(self):
        super(GoTurnRemix, self).__init__()
        # Load an AlexNet model pretrained on ImageNet
        self.features = nn.Sequential(*list(models.alexnet(pretrained=True).children())[:-1])
        # Freeze the pretrained layers
        for param in self.features.parameters():
            param.requires_grad = False
        self.regressor = nn.Sequential(
            nn.Linear(256 * 6 * 6 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4),
        )
        # Initialize the biases of the Linear layers to 1
        # Initialize weights to a normal distribution with 0 mean and 0.005 standard deviation
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(1)
                m.weight.data.normal_(0, 0.005)

    def forward(self, previous, current):
        previous_features = self.features(previous)
        current_features = self.features(current)
        # Flatten, concatenate and pass to regressor the features
        return self.regressor(torch.cat((previous_features.view(previous_features.size(0), 256 * 6 * 6),
                                          current_features.view(current_features.size(0), 256 * 6 * 6)), 1))

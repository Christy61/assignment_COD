import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FeatureCamouflageNet(nn.Module):
    def __init__(self, feature_dim=512, feature_size=(12, 12)):
        super(FeatureCamouflageNet, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        """
        features: (batch_size, 512, H/32, W/32)
        """
        output = self.decoder(features)  # (batch_size, 1, H, W)
        return self.sigmoid(output)

def entropy_loss(outputs, targets):
    EPS = 1e-10
    loss = - (targets @ torch.log(outputs + EPS)
              + (1 - targets) @ torch.log(1 - outputs + EPS)) / targets.shape[0]
    return loss
import torch  # noqa
import torch.nn.functional as F  # noqa
from torch import nn, Tensor  # noqa
import torch
from torch.autograd import Variable
import numpy as np

# Source: https://github.com/berdakh/ERDS-Pytorch/blob/master/nu_models.py
class ERDS_EEGNet(nn.Module):
    def __init__(self,
                 time_samples,
                 channels):
        super(ERDS_EEGNet, self).__init__()

        self.T = time_samples
        self.chans = channels
        self.in_size = (1, time_samples, channels)

        self.layer1 = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 16, (1, self.chans), padding=0),
            nn.BatchNorm2d(16, False),
            nn.ELU(),
            nn.Dropout(0.25))

        self.layer2and3 = nn.Sequential(
            # Layer 2
            nn.ZeroPad2d((16, 17, 0, 1)),
            nn.Conv2d(1, 4, (2, 32)),
            nn.BatchNorm2d(4, False),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 4),

            # Layer 3
            nn.ZeroPad2d((2, 1, 4, 3)),
            nn.Conv2d(4, 4, (8, 4)),
            nn.BatchNorm2d(4, False),
            nn.Dropout(0.25),
            nn.MaxPool2d((2, 4)))

        self.flat_fts = self.get_out_dim(self.in_size)
        self.fc1 = nn.Linear(self.flat_fts, 2)

    def get_out_dim(self, in_size):
        with torch.no_grad():
            # create a tensor
            x = Variable(torch.ones(1, *self.in_size))
            x = self.layer1(x)
            x = x.permute(0, 3, 1, 2)
            x = self.layer2and3(x)
            x = int(np.prod(x.size()[1:]))
            return x

    def forward(self, x):
        x = self.layer1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.layer2and3(x)
        x = x.view(-1, self.flat_fts)
        x = self.fc1(x)
        return x

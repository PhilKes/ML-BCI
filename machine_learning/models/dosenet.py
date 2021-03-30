# Keras model Source:
# https://github.com/hauke-d/cnn-eeg/blob/master/models.py

import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
import numpy as np

from machine_learning.models.eegnet import get_padding


class DoseNet(nn.Module):
    def __init__(self, C, n_class, T, l1=0):
        super(DoseNet, self).__init__()
        # self.fc21 = nn.Linear(in_count, 50)
        # self.fc323= nn.Linear(50, 25)
        # self.fc343 = nn.Linear(25, out_count)
        # self.softmax = nn.Softmax(dim=1)
        # self.con33v2 = nn.Conv2d(F1, D * F1, (C, 1), groups=F1, bias=False)
        input_shape = (T, C, 1)
        # TODO l1: weight_decay in optim.Adam(weight_decay=l1)
        kernel_size1 = (30, 1)
        self.conv1_pad = nn.ZeroPad2d(get_padding(kernel_size1))
        self.conv1 = nn.Conv2d(out_channels=40,
                               in_channels=T,
                               kernel_size=kernel_size1)
        kernel_size2 = (1, C)
        # TODO Padding valid
        self.conv2 = nn.Conv2d(out_channels=40,
                               in_channels=40,
                               kernel_size=kernel_size2)

        self.pool1 = nn.AvgPool2d((30, 1), stride=(15, 1))
        in1 = 1
        in2 = 2
        self.fc1 = nn.Linear(in1, 80)
        self.fc2 = nn.Linear(in2, n_class)

    def forward(self, x):
        x = x.permute(0,3,2,1)
        x = self.conv1_pad(x)
        x = F.relu(self.conv1(x))
        # input shape:
        # (batch_size, samples, chs, 1)
        # After conv1:
        # (batch_size, 40, chs, 1)

        # TODO Pad2
        x = F.relu(self.conv2(x))
        x = self.pool1(1)
        # Flatten
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa


# Model
# see https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb


class EEGNet(nn.Module):
    def __init__(self, n_classes, channels):
        super(EEGNet, self).__init__()
        # self.T = 160

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, channels), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # max pooling sizes x 80
        self.fc1 = nn.Linear(2 * 4 * 40, n_classes)

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1, 2 * 4 * 40)
        # Original: x = F.sigmoid(self.fc1(x))
        # For BCELoss:
        # x = torch.sigmoid(self.fc1(x))
        x = self.fc1(x)
        return x

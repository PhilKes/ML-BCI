import math
from functools import reduce
from operator import __add__
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa

from config import SAMPLES


def get_padding(kernel_size):
    return reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])


# see https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb
# original paper: https://arxiv.org/pdf/1611.08024.pdf

# Input Shape: (Channels, Timepoints, 1)
class EEGNetv2(nn.Module):
    def __init__(self, n_classes, channels, samples=SAMPLES, dropoutRate=0.25, F1=16, D=2, poolLength=8,
                 kernelLength=80):
        super(EEGNetv2, self).__init__()
        self.srate = 160
        self.samples = samples
        self.dropoutRate = dropoutRate

        # https://stackoverflow.com/questions/58307036/is-there-really-no-padding-same-option-for-pytorchs-conv2d
        kernel_size1 = (1, kernelLength)

        # Layer 1
        self.padding1 = nn.ZeroPad2d(get_padding(kernel_size1))
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=F1, kernel_size=kernel_size1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(num_features=F1)
        F2 = 4

        # Layer 2
        kernel_size2 = (channels, 1)
        self.conv2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=kernel_size2)

        self.batchnorm2 = nn.BatchNorm2d(num_features=F2)
        self.pooling2 = nn.AvgPool2d((1, poolLength))

        # Layer 3
        kernel_size3 = (1, 16)
        self.padding2 = nn.ZeroPad2d(get_padding(kernel_size3))
        self.conv3 = nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=kernel_size3)
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        self.pooling3 = nn.AvgPool2d((1, 8))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        self.fc1 = nn.Linear(int(self.samples / 2), n_classes)

    def forward(self, x):
        # Layer 1
        # x = F.elu(self.conv1(x))
        x = self.padding1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        # x = F.dropout(x, self.dropoutRate)
        x = x.permute(0, 3, 1, 2)

        # print("after L1:",x.shape)

        # Layer 2
        x = self.conv2(x)
        x = F.elu(self.batchnorm2(x))
        x = self.pooling2(x)
        x = F.dropout(x, self.dropoutRate)

        # print("after L2:",x.shape)

        # Layer 3
        x = self.padding2(x)
        x = self.conv3(x)
        x = F.elu(self.batchnorm3(x))
        x = self.pooling3(x)
        x = F.dropout(x, self.dropoutRate)

        # print("after L3:",x.shape)

        # FC Layer
        x = x.view(-1, int(self.samples / 2))
        # Original: x = F.sigmoid(self.fc1(x))
        # For BCELoss:
        # x = torch.sigmoid(self.fc1(x))
        x = self.fc1(x)
        # print("after FC:",x.shape)
        return x

# Evaluate function returns values of different criteria like accuracy, precision etc.
# In case you face memory overflow issues, use batch size to control how many samples get
# evaluated at one time. Use a batch_size that is a factor of length of samples.
# This ensures that you won't miss any samples.
# def evaluate(model, X, Y, params=["acc"]):
#     results = []
#     batch_size = 69
#
#     predicted = []
#
#     for i in range(int(len(X) / batch_size)):
#         s = i * batch_size
#         e = i * batch_size + batch_size
#
#         inputs = Variable(torch.from_numpy(X[s:e]))
#         pred = model(inputs)
#
#         predicted.append(pred.data.cpu().numpy())
#
#     inputs = Variable(torch.from_numpy(X))
#     predicted = model(inputs)
#
#     predicted = predicted.data.cpu().numpy()
#
#     for param in params:
#         if param == 'acc':
#             results.append(accuracy_score(Y, np.round(predicted)))
#         if param == "auc":
#             results.append(roc_auc_score(Y, predicted))
#         if param == "recall":
#             results.append(recall_score(Y, np.round(predicted)))
#         if param == "precision":
#             results.append(precision_score(Y, np.round(predicted)))
#         if param == "fmeasure":
#             precision = precision_score(Y, np.round(predicted))
#             recall = recall_score(Y, np.round(predicted))
#             results.append(2 * precision * recall / (precision + recall))
#     return results

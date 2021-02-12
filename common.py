"""
Main functions for
Training, Testing, Benchmarking
"""
import math
import sys
import time

import numpy as np
import torch  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch import nn, Tensor  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa
from tqdm import tqdm

from config import LR


# Training
# see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network
def train(net, data_loader, epochs=1, device=torch.device("cpu"), lr=LR):
    net.train()
    # Init Loss Function + Optimizer with Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr['start'])
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr['milestones'], gamma=lr['gamma'])
    print("###### Training started")
    loss_values = np.zeros((epochs))
    for epoch in range(epochs):
        print(f"## Epoch {epoch} ")
        running_loss = 0.0
        # Wrap in tqdm for Progressbar in Console
        pbar = tqdm(data_loader, file=sys.stdout)
        # Training in batches from the DataLoader
        for idx_batch, (inputs, labels) in enumerate(pbar):
            # Convert to correct types + put on GPU
            inputs, labels = inputs, labels.long().to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        pbar.close()
        lr_scheduler.step()
        # Loss of entire epoch / amount of batches
        loss_values[epoch] = (running_loss / len(data_loader))
        print('[%3d] Total loss: %f' %
              (epoch, running_loss))
        if math.isnan(loss_values[epoch]):
            print("loss_values[epoch]", loss_values[epoch])
            print("running_loss", running_loss)
            print("len(data_loader)", len(data_loader))
            break
    print("Training finished ######")

    return loss_values


# Tests labeled data with trained net
def test(net, data_loader, device=torch.device("cpu")):
    print("###### Testing started")
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        net.eval()
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs, labels.float()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data.cpu(), 1)
            # For BCELOSS:
            #   revert np.eye -> [0 0 1] = 2, [1 0 0] = 0
            #   labels = np.argmax(labels.cpu(), axis=1)
            labels = labels.cpu()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = (100 * correct / total)
    print(F'Accuracy on the {len(data_loader.dataset)} test trials: %0.2f %%' % (
        acc))
    print("Testing finished ######")
    return acc


# Benchmarks net on Inference Time in Batches
def benchmark(net, data_loader, device=torch.device("cpu"), fp16=False):
    print("###### Inference started")
    with torch.no_grad():
        net.eval()
        start = time.perf_counter()
        num_batches = len(data_loader)
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.float()
            outputs = net(inputs.half() if fp16 else inputs)
            print(f"Batch: {i + 1} of {num_batches}")
        stop = time.perf_counter()
        # Latency of one batch
        print(f"Batches:{num_batches}")
        batch_lat = (stop - start) / num_batches
        # Inference time for 1 Trial
        print(f"Trials:{len(data_loader.dataset)}")
        trial_inf_time = (stop - start) / len(data_loader.dataset)

    print("Inference finished ######")
    return (batch_lat, trial_inf_time)

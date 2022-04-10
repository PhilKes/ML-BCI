"""
Main functions for
Inferencing, Training, Testing, Benchmarking
"""
import logging
import sys
import time

import numpy as np
import torch  # noqa
import torch.optim as optim  # noqa
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication
from sklearn.metrics import accuracy_score
from torch import nn  # noqa
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch as t
import torch.types

from app.config import CONFIG
from app.machine_learning.configs_results import benchmark_single_result_str
from app.machine_learning.util import get_class_accuracies
from app.util.progress_wrapper import TqdmProgressBar


def do_train(model: t.nn.Module, loader_train: DataLoader, loader_valid: DataLoader, epochs: int = 1,
             device: t.types.Device = CONFIG.DEVICE, early_stop=False, qthread : QThread=None):
    """
    Training
    see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network

    In every epoch: After Inference + Calc Loss + Backpropagation on Test Dataset
    a Test Dataset of 1 Subject is used to calculate test_loss on trained model of current epoch
    to determine best model with loweset test_loss
    :return:
    loss_values_train: Loss value of every Epoch on Training Dataset (data_loader)
    loss_values_valid: Loss value of every Epoch on Test Dataset (loader_test_loss)
    best_model: state_dict() of epoch model with lowest test_loss if early_stop=True
    best_epoch: best_epoch with lowest test_loss if early_stop=True
    """
    model.train()
    # Init Loss Function + Optimizer with Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.MI.LR.start)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CONFIG.MI.LR.milestones,
                                                  gamma=CONFIG.MI.LR.gamma)
    logging.info("###### Training started")
    loss_values_train, loss_values_valid = np.full((epochs), fill_value=np.inf), np.full((epochs), fill_value=np.inf)
    best_epoch = 0
    best_model = model.state_dict().copy()
    for epoch in range(epochs):
        logging.info(f"## Epoch {epoch} ")
        model.train()
        running_loss_train, running_loss_valid = 0.0, 0.0
        # Wrap in tqdm for Progressbar in Console
        pbar = TqdmProgressBar(loader_train)
        # Training in batches from the DataLoader
        for idx_batch, (inputs, labels) in enumerate(pbar):
            # Convert to correct types + put on GPU
            inputs, labels = inputs, labels.long().to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            # logging.info("out %s labels %s",outputs.shape,labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()
        pbar.close()
        # Check if thread was stopped
        if qthread is not None:
            QApplication.processEvents()
            if qthread.isInterruptionRequested() or not qthread.isRunning():
                return loss_values_train, loss_values_valid, best_model, best_epoch
        # Loss of entire epoch / amount of batches
        epoch_loss_train = (running_loss_train / len(loader_train))
        if loader_valid is not None:
            # Validation loss on Test Dataset
            # if early_stop=True: Used to determine best model state
            with torch.no_grad():
                model.eval()
                for idx_batch, (inputs, labels) in enumerate(loader_valid):
                    # Convert to correct types + put on GPU
                    inputs, labels = inputs, labels.long().to(device)
                    # zero the parameter gradients
                    # optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = model(inputs)
                    # logging.info("out %s labels %s",outputs.shape,labels.shape)
                    loss = criterion(outputs, labels)
                    # loss.backward()
                    # optimizer.step()
                    running_loss_valid += loss.item()
            epoch_loss_valid = (running_loss_valid / len(loader_valid))
            # Determine if epoch (validation loss) is lower than all epochs before -> current best model
            if early_stop:
                if epoch_loss_valid < loss_values_valid.min():
                    best_model = model.state_dict().copy()
                    best_epoch = epoch
            logging.info('[%3d] Training loss/batch: %f\tTesting loss/batch: %f' %
                  (epoch, epoch_loss_train, epoch_loss_valid))
            loss_values_valid[epoch] = epoch_loss_valid
        else:
            logging.info('[%3d] Training loss/batch: %f' ,epoch, epoch_loss_train)
        loss_values_train[epoch] = epoch_loss_train

        lr_scheduler.step()
    logging.info("Training finished ######")

    return loss_values_train, loss_values_valid, best_model, best_epoch


def do_test(model: t.nn.Module, data_loader: DataLoader) -> (float, np.ndarray, np.ndarray):
    """
    Tests labeled data with trained net
    :return:
    acc: Average accuracy across entire data_loader
    act_labels: Actual Labels of the Trials
    pred_labels: Predicted Labels of the Trials
    """
    logging.info("###### Testing started")
    act_labels = np.zeros((len(data_loader.dataset)), dtype=np.int)
    pred_labels = np.zeros((len(data_loader.dataset)), dtype=np.int)
    sample_idx = 0
    with torch.no_grad():
        model.eval()
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs, labels.float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data.cpu(), 1)
            # For BCELOSS:
            #   revert np.eye -> [0 0 1] = 2, [1 0 0] = 0
            #   labels = np.argmax(labels.cpu(), axis=1)
            labels = labels.cpu()
            # Get actual and predicted label for current batch
            for (pred, label) in zip(predicted, labels):
                pred, label = int(pred.item()), int(label.item())
                act_labels[sample_idx] = label
                pred_labels[sample_idx] = pred
                sample_idx += 1

    acc = 100 * accuracy_score(act_labels, pred_labels)
    logging.info(F'Accuracy on the {len(data_loader.dataset)} test trials: %0.2f %%' % (
        acc))
    logging.info(F'Class Accuracies: {get_class_accuracies(act_labels, pred_labels)}')
    logging.info("Testing finished ######")
    return acc, act_labels, pred_labels


def do_benchmark(model: t.nn.Module, data_loader: DataLoader, device=CONFIG.DEVICE, fp16=False) -> (
        float, float, float):
    """
    Benchmarks model on Inference Time in Batches
    :return:
    batch_lat: Batch Latency
    trial_inf_time: Trial Inference Time
    acc: Achieved Accuracy
    """
    # TODO Best way to measure timings? (w/ device= Cuda/Cpu)
    # INIT LOGGERS
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    logging.info("###### Inference started")
    model.eval()
    num_batches = len(data_loader)
    # https://deci.ai/the-correct-way-to-measure-inference-time-of-deep-neural-networks/?utm_referrer=https%3A%2F%2Fwww.google.com%2F
    timings = np.zeros((num_batches))
    total, correct = 0.0, 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            if fp16:
                inputs = inputs.half()
            # starter.record()
            start = time.perf_counter()
            outputs = model(inputs)
            # ender.record()
            # WAIT FOR GPU SYNC
            # torch.cuda.synchronize()
            stop = time.perf_counter()
            outputs = outputs.float()
            _, predicted = torch.max(outputs.data.cpu(), 1)
            labels = labels.cpu()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # curr_time = starter.elapsed_time(ender)
            curr_time = stop - start
            timings[i] = curr_time

        acc = (100 * correct / total)
        batch_lat = np.sum(timings) / num_batches
        trials = len(data_loader.dataset)
        trial_inf_time = np.sum(timings) / trials
        logging.info(benchmark_single_result_str(trials, acc, num_batches, batch_lat, trial_inf_time))

        # trial_inf_time = (stop - start) / len(data_loader.dataset)

    logging.info("Inference finished ######")
    return batch_lat, trial_inf_time, acc


def do_predict_on_samples(model: t.nn.Module, n_class: int, samples_data: np.ndarray, max_sample: int,
                          device=CONFIG.DEVICE):
    """
    Infers on all given samples with time window
    Returns all class predictions for all samples
    :param samples_data: Numpy Array containing EEG Samples with Shape (Channel, Sample)
    :param max_sample: Number of last sample to predict up to
    """
    sample_predictions = np.zeros((max_sample, n_class))
    logging.info('Predicting on every sample of run')

    pbar = TqdmProgressBar(range(max_sample))
    for now_sample in pbar:
        if now_sample < CONFIG.EEG.SAMPLES:
            continue
        # label, now_time = get_label_at_idx(times, raw.annotations, now_sample)
        time_window_samples = samples_data[:, (now_sample - CONFIG.EEG.SAMPLES):now_sample]
        sample_predictions[now_sample] = do_predict_single(model, time_window_samples, device)
    return sample_predictions


def do_predict_single(model: t.nn.Module, X: np.ndarray, device=CONFIG.DEVICE):
    """
    Infers on single Trial (SAMPLES)
    Returns class predictions (with Softmax -> predictions =[0;1])
    :param X: Single Trial Data Array with Shape (Channel, Sample)
    """
    with torch.no_grad():
        X = torch.as_tensor(X[None, None, ...], device=device, dtype=torch.float32)
        output = model(X)
        # logging.info("Out %s",output)
        # _, predicted = torch.max(output.data.cpu(), 1)
        # predicted = F.softmax(output, dim=1)
        predicted = output
    return predicted.cpu()

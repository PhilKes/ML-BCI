#!/usr/bin/python3
"""
Main Python script to execute
"""
import argparse

import torch

from EEGNet_physionet import eegnet_training_cv
from config import EPOCHS, CUDA


def main():
    parser = argparse.ArgumentParser(description=' Main script to run either Training (+Cross Validation) or Benchmarking'
                                                 ' of EEGNet on Physionet Motor Imagery Dataset')
    parser.add_argument('-train',
                        help="Runs Training with Cross Validation with Physionet Dataset",
                        action='store_true', required=False)
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of epochs for Training (default:{EPOCHS})')
    parser.add_argument('--n_classes', nargs='+', type=int, default=[3],
                        help="List of n-class Classifications to run (2/3/4-Class possible)")

    parser.add_argument('-benchmark', help="Runs Benchmarking with Physionet Dataset",
                        action='store_true', required=False)

    parser.add_argument('--loops', type=int, default=1,
                        help=f'Number of times Training/Benchmarking is run (default:1)')

    args = parser.parse_args()
    print(args)
    if (not args.train) & (not args.benchmark):
        parser.error("Either flag '--train' or '--benchmark' must be present!")
    if not all(((n_class >= 2) & (n_class <= 4)) for n_class in args.n_classes):
        parser.error("Invalid n-class Classification specified (2/3/4-Class possible)")

    # Use GPU for model & tensors if available
    dev = None
    if CUDA & torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    if args.train:
        for i in range(args.loops):
            eegnet_training_cv(num_epochs=args.epochs, device=device)
    elif args.benchmark:
        for i in range(args.loops):
            print("Benchmarking")


########################################################################################
if __name__ == '__main__':
    main()

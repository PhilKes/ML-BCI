#!/usr/bin/python3
"""
Main Python script to execute either Training with Cross Validation on Physionet Dataset
or Benchmarking of Inference (Batch Latency + Inference time per Trial)
"""
import argparse

import torch

from EEGNet_physionet import eegnet_training_cv, eegnet_benchmark
from config import EPOCHS, CUDA, SUBJECTS_CS
from data_loading import ALL_SUBJECTS


def main():
    parser = argparse.ArgumentParser(
        description=' Main script to run either Training (+Cross Validation) or Benchmarking'
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
    # If DATA_PRELOAD=True (config.py): high memory usage -> decrease subjects for lower memory usage when benchmarking
    parser.add_argument('--subjects_cs', type=int, default=SUBJECTS_CS,
                        help=f"Chunk size for preloading subjects in memory (only for benchmark, default:{SUBJECTS_CS}, lower for less memory usage )")
    parser.add_argument('--trt', action='store_true', required=False,
                        help=f"Use TensorRT to optimize trained EEGNet")
    parser.add_argument('--fp16', action='store_true', required=False,
                        help=f"Use fp16 for TensorRT optimization")
    parser.add_argument('--iters', type=int, default=1,
                        help=f'Number of benchmark iterations over the Dataset in a loop (default:1)')

    parser.add_argument('--loops', type=int, default=1,
                        help=f'Number of loops of Training/Benchmarking is run (default:1)')

    args = parser.parse_args()
    print(args)
    if (not args.train) & (not args.benchmark):
        parser.error("Either flag '--train' or '--benchmark' must be present!")
    if not all(((n_class >= 2) & (n_class <= 4)) for n_class in args.n_classes):
        parser.error("Invalid n-class Classification specified (2/3/4-Class possible)")
    if args.subjects_cs > len(ALL_SUBJECTS):
        parser.error(f"Maximum subjects_bs: {len(ALL_SUBJECTS)}")
    if (args.iters > 1) & (not args.benchmark):
        parser.error(f"Iteration parameter is only used if benchmarking")
    if args.fp16 & (not args.trt):
        parser.error(f"Floating Point16 only availabe if TensorRT optimization is enabled too")

    # Use GPU for model & tensors if available
    dev = None
    if CUDA & torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    if args.train:
        for i in range(args.loops):
            eegnet_training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes)
    elif args.benchmark:
        for i in range(args.loops):
            # For now only 3-Class Classification for benchmarking
            eegnet_benchmark(n_classes=[3], device=device, subjects_cs=args.subjects_cs,
                             tensorRT=args.trt,fp16=args.fp16)


########################################################################################
if __name__ == '__main__':
    main()

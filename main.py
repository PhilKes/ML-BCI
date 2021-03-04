#!/usr/bin/python3
"""
Main Python script to execute either Training with Cross Validation on Physionet Dataset (-train)
or Benchmarking of Inference (Batch Latency + Inference time per Trial) (-benchmark)
Configuration Parameters for Number of Epochs, TensorRT Optimizations,... (see main.py --help)
"""
import argparse
from datetime import datetime
import sys

import torch

from EEGNet_physionet import eegnet_training_cv, eegnet_benchmark
from config import EPOCHS, SUBJECTS_CS, BATCH_SIZE, CHANNELS, MNE_CHANNELS, MOTORIMAGERY_CHANNELS
from data_loading import ALL_SUBJECTS
from utils import datetime_to_folder_str


def single_run(argv=sys.argv[1:]):
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
    parser.add_argument('--ch_names', nargs='+', type=str, default=MNE_CHANNELS,
                        help="List of EEG Channels to use (see config.py MNE_CHANNELS for all available Channels)")
    parser.add_argument('--ch_motorimg', action='store_true',
                        help=f"Use Predefined Motor Imagery Channels for Training ({MOTORIMAGERY_CHANNELS})")

    parser.add_argument('-benchmark',
                        help="Runs Benchmarking with Physionet Dataset with trained model (./benchmarking_model/trained_model.pt)",
                        action='store_true', required=False)
    # If DATA_PRELOAD=True (config.py): high memory usage -> decrease subjects for lower memory usage when benchmarking
    parser.add_argument('--subjects_cs', type=int, default=SUBJECTS_CS,
                        help=f"Chunk size for preloading subjects in memory (only for benchmark, default:{SUBJECTS_CS}, lower for less memory usage )")
    parser.add_argument('--trt', action='store_true',
                        help=f"Use TensorRT to optimize trained EEGNet")
    parser.add_argument('--fp16', action='store_true',
                        help=f"Use fp16 for TensorRT optimization")
    parser.add_argument('--iters', type=int, default=1,
                        help=f'Number of benchmark iterations over the Dataset in a loop (default:1)')
    parser.add_argument('--bs', type=int, default=BATCH_SIZE,
                        help=f'Trial Batch Size (default:{BATCH_SIZE})')

    # parser.add_argument('--loops', type=int, default=1,
    #                     help=f'Number of loops of Training/Benchmarking is run (default:1)')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for executed Run (stores results in ./results/{benchmark/training}/{name})')
    parser.add_argument('--tag', type=str, default=None, required=False,
                        help='Optional Tag for results files (e.g. for different batch sizes)')
    parser.add_argument('--device', type=str, default="gpu", required=False,
                        help='Either "gpu" or "cpu"')

    args = parser.parse_args(argv)
    print(args)
    if args.ch_motorimg:
        args.ch_names = MOTORIMAGERY_CHANNELS
        if args.name is None:
            start = datetime.now()
            args.name = f"{datetime_to_folder_str(start)}_motor_img"
        else:
            args.name = args.tag + "_motor_img"
    if (not args.train) & (not args.benchmark):
        parser.error("Either flag '--train' or '--benchmark' must be present!")
    if not all(((n_class >= 2) & (n_class <= 4)) for n_class in args.n_classes):
        parser.error("Invalid n-class Classification specified (2/3/4-Class possible)")
    if args.subjects_cs > len(ALL_SUBJECTS):
        parser.error(f"Maximum subjects_bs: {len(ALL_SUBJECTS)}")
    if (args.iters > 1) & (not args.benchmark):
        parser.error(f"Iteration parameter is only used if benchmarking")
    if args.fp16 & (not args.trt):
        parser.error(f"Floating Point16 only available if TensorRT optimization is enabled too")
    if (args.device != "gpu") & (args.device != "cpu"):
        parser.error(f"Device can either be 'gpu' or 'cpu'")
    if (args.device == "cpu") & (args.trt):
        parser.error(f"Cannot optimize with TensorRT with device='cpu'")
    if (args.device == "cpu") & (args.bs > 15):
        parser.error(f"Cannot use batch size > 15 if device='cpu' (Jetson Nano)")
    if (len(args.ch_names) < 1) | any((ch not in MNE_CHANNELS) for ch in args.ch_names):
        print(args.ch_names)
        parser.error("Channel names (--ch_names) must be a list of EEG Channels (see config.py MNE_CHANNELS)")
    # Slice channels from the 64 available EEG Channels from the start to given chs
    # ch_names = MNE_CHANNELS[:args.chs]

    # Use GPU for model & tensors if available
    dev = None
    if (args.device == "gpu") & torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("device", device.type)

    if args.train:
        # for i in range(args.loops):
        eegnet_training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                           name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names)
    elif args.benchmark:
        # for i in range(args.loops):
        # For now only 3-Class Classification for benchmarking
        return eegnet_benchmark(n_classes=[3], device=device, subjects_cs=args.subjects_cs, name=args.name,
                                tensorRT=args.trt, fp16=args.fp16, iters=args.iters, batch_size=args.bs,
                                tag=args.tag, ch_names=args.ch_names)


########################################################################################
if __name__ == '__main__':
    single_run()

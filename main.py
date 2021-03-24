#!/usr/bin/python3
"""
Main Python script to execute either Training with Cross Validation on Physionet Dataset (-train)
Subject-specific Training (-train-ss)
Benchmarking of Inference (Batch Latency + Inference time per Trial) (-benchmark)
Live-Simulation of a single Subject Run (-live_sim)
Configuration Parameters for Number of Epochs, TensorRT Optimizations,... (see main.py --help)
"""
import argparse
from datetime import datetime
import sys

import mne
import torch

from physionet_machine_learning import physionet_training_cv, physionet_benchmark, physionet_live_sim, \
    physionet_training_ss
from config import EPOCHS, SUBJECTS_CS, BATCH_SIZE, MNE_CHANNELS, MOTORIMG_CHANNELS
from data_loading import ALL_SUBJECTS, excluded_subjects
from util.misc import datetime_to_folder_str, list_to_string, load_chs_of_model


def single_run(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description=' Main script to run either Training (5-Fold CV),Subject-Specific Training, Benchmarking on Inference or Live Simulation'
                    ' of EEG classification with the EEGNet on Physionet Motor Imagery Dataset')
    add_common_arguments(parser)
    add_train_arguments(parser)
    add_train_ss_arguments(parser)
    add_benchmark_arguments(parser)
    add_live_sim_arguments(parser)

    args = parser.parse_args(argv)
    print(args)

    check_common_arguments(parser, args)
    check_train_arguments(parser, args)
    check_train_ss_arguments(parser, args)
    check_benchmark_arguments(parser, args)
    check_live_sim_arguments(parser, args)

    # Dont print MNE loading logs
    mne.set_log_level('WARNING')

    # Use GPU for model & tensors if available
    dev = None
    if (args.device == "gpu") & torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("device", device.type)

    if args.train:
        return physionet_training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                                     name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                                     equal_trials=(not args.all_trials), early_stop=args.early_stop,
                                     excluded=args.excluded)
    elif args.train_ss:
        model_path = f"{args.model}"
        args.ch_names = load_chs_of_model(model_path)
        physionet_training_ss(model_path, args.subject, num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                              name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names)
    elif args.benchmark:
        model_path = f"{args.model}"
        args.ch_names = load_chs_of_model(model_path)
        return physionet_benchmark(model_path, name=args.name, n_classes=args.n_classes, device=device,
                                   subjects_cs=args.subjects_cs,
                                   tensorRT=args.trt, fp16=args.fp16, iters=args.iters, batch_size=args.bs,
                                   tag=args.tag, ch_names=args.ch_names, equal_trials=(not args.all_trials),
                                   continuous=args.continuous)
    elif args.live_sim:
        model_path = f"{args.model}"
        args.ch_names = load_chs_of_model(model_path)
        return physionet_live_sim(model_path, subject=args.subject, name=args.name, ch_names=args.ch_names,
                                  n_classes=args.n_classes, device=device, tag=args.tag,
                                  equal_trials=(not args.all_trials))


# Common Arguments #########################
def add_common_arguments(parser):
    parser.add_argument('--n_classes', nargs='+', type=int, default=[3],
                        help="List of n-class Classifications to run (2/3/4-Class possible)")
    parser.add_argument('--name', type=str, default=None,
                        help='Name for executed Run (stores results in ./results/{benchmark/training}/{name})')
    parser.add_argument('--tag', type=str, default=None, required=False,
                        help='Optional Tag for results files (e.g. for different batch sizes)')
    parser.add_argument('--device', type=str, default="gpu", required=False,
                        help='Either "gpu" or "cpu"')
    parser.add_argument('--bs', type=int, default=BATCH_SIZE,
                        help=f'Trial Batch Size (default:{BATCH_SIZE})')
    parser.add_argument('--model', type=str, default=None,
                        help='Relative Folder path of model used (in ./results folder) for -benchmark or -train_ss')


def check_common_arguments(parser, args):
    if (not args.train) & (not args.benchmark) & (not args.live_sim) & (not args.train_ss):
        parser.error("Either flag '-train', '-train_ss', '-benchmark' or '-live_sim' must be present!")
    if not all(((n_class >= 2) & (n_class <= 4)) for n_class in args.n_classes):
        parser.error("Invalid n-class Classification specified (2/3/4-Class possible)")
    if (args.device != "gpu") & (args.device != "cpu"):
        parser.error(f"Device can either be 'gpu' or 'cpu'")
    if (args.benchmark | args.train_ss) & (args.model is None):
        parser.error("You have to use --model to specify which model to use for -benchmark or -train_ss")
    if (args.device == "cpu") & (args.bs > 15):
        parser.error(f"Cannot use batch size > 15 if device='cpu' (Jetson Nano)")


# Train Arguments #########################
def add_train_arguments(parser):
    parser.add_argument('-train',
                        help="Runs Training with Cross Validation with Physionet Dataset",
                        action='store_true', required=False)
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of epochs for Training (default:{EPOCHS})')
    parser.add_argument('--ch_names', nargs='+', type=str, default=MNE_CHANNELS,
                        help="List of EEG Channels to use (see config.py MNE_CHANNELS for all available Channels)")
    parser.add_argument('--ch_motorimg', type=str, default=None,
                        help=f"Use and set amount of predefined Motor Imagery Channels for Training (either {list_to_string(MOTORIMG_CHANNELS.keys())} channels)")
    parser.add_argument('--all_trials', action='store_true',
                        help=f"Use all available Trials per class for Training (if True, Rest class ('0') has more Trials than other classes)")
    parser.add_argument('--early_stop', action='store_true',
                        help=f'If present, will determine the model with the lowest loss on the validation set')
    parser.add_argument('--excluded', nargs='+', type=int, default=[],
                        help=f'List of Subjects that are excluded during Training (default excluded Subjects:{excluded_subjects})')


def check_train_arguments(parser, args):
    if args.ch_motorimg is not None:
        if args.ch_motorimg not in MOTORIMG_CHANNELS:
            parser.error(
                f"Only {list_to_string(MOTORIMG_CHANNELS.keys())} channels are available for --ch_motorimg option")
            pass
        args.ch_names = MOTORIMG_CHANNELS[args.ch_motorimg]
        if args.name is None:
            start = datetime.now()
            args.name = f"{datetime_to_folder_str(start)}_motor_img{args.ch_motorimg}"
        else:
            args.name = args.name + f"_motor_img{args.ch_motorimg}"
    if (len(args.ch_names) < 1) | any((ch not in MNE_CHANNELS) for ch in args.ch_names):
        print(args.ch_names)
        parser.error("Channel names (--ch_names) must be a list of EEG Channels (see config.py MNE_CHANNELS)")


# Subject-Specific Train Arguments #########
def add_train_ss_arguments(parser):
    parser.add_argument('-train_ss',
                        help="Runs Subject specific Training on pretrained model",
                        action='store_true', required=False)


def check_train_ss_arguments(parser, args):
    pass


# Benchmark Arguments ######################
def add_benchmark_arguments(parser):
    parser.add_argument('-benchmark',
                        help="Runs Benchmarking with Physionet Dataset with specified trained model",
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
    parser.add_argument('--all', dest='continuous', action='store_false',
                        help=f'If present, will only loop benchmarking over entire Physionet Dataset, with loading Subjects chunks in between Inferences (default: False)')


def check_benchmark_arguments(parser, args):
    if args.subjects_cs > len(ALL_SUBJECTS):
        parser.error(f"Maximum subjects_bs: {len(ALL_SUBJECTS)}")
    if (args.iters > 1) & (not args.benchmark):
        parser.error(f"Iteration parameter is only used if benchmarking")
    if args.fp16 & (not args.trt):
        parser.error(f"Floating Point16 only available if TensorRT optimization is enabled too")
    if (args.device == "cpu") & (args.trt):
        parser.error(f"Cannot optimize with TensorRT with device='cpu'")


# Live-Simulation Arguments ################
def add_live_sim_arguments(parser):
    parser.add_argument('-live_sim',
                        help="Simulate live usage of a subject with n_class classification on 1 single run",
                        action='store_true', required=False)
    parser.add_argument('--subject', type=int, default=1,
                        help=f'Subject used for -live_sim or -train_ss')


def check_live_sim_arguments(parser, args):
    if args.live_sim & (args.subject not in ALL_SUBJECTS):
        parser.error(f"Subject {args.subject} does not exist!")


########################################################################################
if __name__ == '__main__':
    single_run()

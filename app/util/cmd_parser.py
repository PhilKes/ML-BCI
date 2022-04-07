"""
File: cmd_parser.py

Helper methods to implement Command Line Interface(CLI)
providing arguments for all relevant Parameters for every Mode


Author:
  Originally developed by Philipp Kessler (May 2021)

History:
  2021-05-10: Common parameter mi_ds included - ms (Manfred Strahnen)
"""

import argparse
import logging

from app.config import CONFIG, MOTORIMG_CHANNELS
from app.data.datasets.datasets import DATASETS
from app.data.datasets.phys.phys_dataset import PHYS
from app.defaults import DEFAULT_PARAMS
from app.util.misc import list_to_str


def create_parser():
    parser = argparse.ArgumentParser(
        description=' Main script to run either Training (5-Fold CV),Subject-Specific Training, Benchmarking on Inference or Live Simulation'
                    ' of EEG classification with the EEGNet on the Physionet Motor Imagery Dataset')
    add_common_arguments(parser)
    add_train_arguments(parser)
    add_train_ss_arguments(parser)
    add_benchmark_arguments(parser)
    add_live_sim_arguments(parser)
    parser.add_argument('--no-gui', action='store_true')
    return parser


def parse_and_check(parser, argv, check=True):
    args = parser.parse_args(argv)
    logging.info(args)
    if check:
        check_common_arguments(parser, args)
        check_train_arguments(parser, args)
        check_train_ss_arguments(parser, args)
        check_benchmark_arguments(parser, args)
        check_live_sim_arguments(parser, args)
    return args


# Common Arguments #########################
def add_common_arguments(parser):
    parser.add_argument('--n_classes', nargs='+', type=int, default=DEFAULT_PARAMS.n_classes,
                        help="List of n-class Classifications to run (2/3/4-Class possible)")
    parser.add_argument('--name', type=str, default=DEFAULT_PARAMS.name,
                        help='Name for executed Run (stores results in ./results/{benchmark/training}/{name})')
    parser.add_argument('--tag', type=str, default=DEFAULT_PARAMS.tag, required=False,
                        help='Optional Tag for results files (e.g. for different batch sizes)')
    parser.add_argument('--device', type=str, default=DEFAULT_PARAMS.device, required=False,
                        help='Device to use, either "gpu" or "cpu"')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_PARAMS.batch_size,
                        help=f'Trial Batch Size (default:{DEFAULT_PARAMS.batch_size})')
    parser.add_argument('--model', type=str, default=DEFAULT_PARAMS.model,
                        help='Relative Folder path of trained model(in ./results/.../training/ folder), used for -benchmark or -train_ss or -live_sim')
    parser.add_argument('--subject', type=int, default=DEFAULT_PARAMS.subject,
                        help=f'Subject used for -live_sim or -train_ss')

    parser.add_argument('--trials_slices', type=int, default=DEFAULT_PARAMS.trials_slices,
                        help=f'Will slice every Trial into n x Trials (default: {DEFAULT_PARAMS.trials_slices})')
    parser.add_argument('--tmin', type=float, default=DEFAULT_PARAMS.tmin,
                        help=f'Start Time of every Trial Epoch')
    parser.add_argument('--tmax', type=float, default=DEFAULT_PARAMS.tmax,
                        help=f'End Time of every Trial Epoch')
    parser.add_argument('--dataset', type=str, default=DEFAULT_PARAMS.dataset,
                        help=f'Name of the MI dataset (available: {",".join([ds for ds in DATASETS])})')


def check_common_arguments(parser, args):
    if (not args.train) & (not args.benchmark) & (not args.live_sim) & (not args.train_ss):
        parser.error("Either flag '-train', '-train_ss', '-benchmark' or '-live_sim'  must be present!")
    if not all(((n_class >= 2) & (n_class <= 4)) for n_class in args.n_classes):
        parser.error("Invalid n-class Classification specified (2/3/4-Class possible)")
    if (args.device != "gpu") & (args.device != "cpu"):
        parser.error(f"Device can either be 'gpu' or 'cpu'")
    if (args.benchmark | args.train_ss) & (args.model is None):
        parser.error("You have to use --model to specify which model to use for -benchmark or -train_ss")
    if (args.device == "cpu") & (args.batch_size > 15):
        parser.error(f"Cannot use batch size > 15 if device='cpu' (Jetson Nano)")
    if (args.live_sim | args.train_ss) & (args.subject is not None) & (args.subject not in PHYS.ALL_SUBJECTS):
        parser.error(f"Subject {args.subject} does not exist!")

    if args.dataset not in DATASETS:
        parser.error(f"Dataset '{args.dataset}' does not exist (available: {','.join([ds for ds in DATASETS])}))")

    dataset = DATASETS[args.dataset]

    # Adjust global parameters which depend on the selected dataset
    if (args.ch_names == None) & (args.ch_motorimg == None):
        args.ch_names = dataset.CONSTANTS.CHANNELS
    if ((args.tmin != None) and (args.tmax == None)) or ((args.tmin == None) and (args.tmax != None)):
        parser.error("You have to either set the 'tmax' AND 'tmin' options or none of the two options")
    # Dataset dependent EEG config structure re-initialization
    CONFIG.set_eeg_config(dataset.CONSTANTS.CONFIG)
    if (args.tmin is not None) and (args.tmax is not None):
        if (args.tmin > args.tmax) or (args.tmin == args.tmax):
            parser.error(f"tmax has to be greater than tmin!")
        else:
            CONFIG.EEG.set_times(args.tmin, args.tmax, CONFIG.EEG.CUE_OFFSET)
    if args.trials_slices < 1:
        parser.error(f"Trials slices has to be greater than 0!")
    if (CONFIG.EEG.SAMPLES % args.trials_slices != 0):
        parser.error(f"Can't divide {CONFIG.EEG.SAMPLES} Samples in {args.trials_slices} slices!")
    CONFIG.EEG.set_trials_slices(args.trials_slices)


# Train Arguments #########################
def add_train_arguments(parser):
    parser.add_argument('-train',
                        help="Runs Training with Cross Validation with Physionet Dataset",
                        action='store_true', required=False)
    parser.add_argument('--epochs', type=int, default=DEFAULT_PARAMS.epochs,
                        help=f'Number of epochs for Training (default:{DEFAULT_PARAMS.epochs})')
    parser.add_argument('--ch_names', nargs='+', type=str, default=DEFAULT_PARAMS.ch_names,
                        help="List of EEG Channels to use")
    parser.add_argument('--ch_motorimg', type=str, default=DEFAULT_PARAMS.ch_motorimg,
                        help=f"""Use and set amount of predefined Motor Imagery Channels for Training (either {list_to_str(
                            MOTORIMG_CHANNELS.keys())} channels""")
    parser.add_argument('--all_trials', action='store_false' if DEFAULT_PARAMS.all_trials is True else 'store_true',
                        help=f"Use all available Trials per class for Training (if True, Rest class ('0') has more Trials than other classes)")
    parser.add_argument('--early_stop', action='store_false' if DEFAULT_PARAMS.early_stop is True else 'store_true',
                        help=f'If present, will determine the model with the lowest loss on the validation set')
    parser.add_argument('--excluded', nargs='+', type=int, default=DEFAULT_PARAMS.excluded,
                        help=f'List of Subjects that are excluded during Training')

    parser.add_argument('--only_fold', type=int, default=DEFAULT_PARAMS.only_fold,
                        help=f'Optional: Specify single Fold to be only trained on')


def check_train_arguments(parser, args):
    if args.ch_motorimg is not None:
        # TODO Implement for BCIC
        if args.ch_motorimg not in MOTORIMG_CHANNELS:
            parser.error(
                f"Only {list_to_str(MOTORIMG_CHANNELS.keys())} channels are available for --ch_motorimg option")
            pass
        args.ch_names = MOTORIMG_CHANNELS[args.ch_motorimg]


#    if (len(args.ch_names) < 1) | any((ch not in MNE_CHANNELS) for ch in args.ch_names):
#        logging.info(args.ch_names)
#        parser.error("Channel names (--ch_names) must be a list of EEG Channels (see config.py MNE_CHANNELS)")


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
    parser.add_argument('--subjects_cs', type=int, default=DEFAULT_PARAMS.subjects_cs,
                        help=f"Chunk size for preloading subjects in memory (only for benchmark, default:{CONFIG.MI.SUBJECTS_CS}, lower for less memory usage )")
    parser.add_argument('--trt', action='store_false' if DEFAULT_PARAMS.trt is True else 'store_true',
                        help=f"Use TensorRT to optimize trained EEGNet")
    parser.add_argument('--fp16', action='store_false' if DEFAULT_PARAMS.fp16 is True else 'store_true',
                        help=f"Use fp16 for TensorRT optimization")
    parser.add_argument('--iters', type=int, default=DEFAULT_PARAMS.iters,
                        help=f'Number of benchmark iterations over the Dataset in a loop (default:1)')
    parser.add_argument('--all', dest='continuous',
                        action='store_false' if DEFAULT_PARAMS.all is True else 'store_true',
                        help=f'If present, will only loop benchmarking over entire Physionet Dataset, with loading Subjects chunks in between Inferences (default: False)')


def check_benchmark_arguments(parser, args):
    if args.subjects_cs > len(PHYS.ALL_SUBJECTS):
        parser.error(f"Maximum subjects_bs: {len(PHYS.ALL_SUBJECTS)}")
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


def check_live_sim_arguments(parser, args):
    pass

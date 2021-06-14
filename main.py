#!/usr/bin/python3
"""
Main Python script to execute either Training with Cross Validation on Physionet Dataset (-train)
Subject-specific Training (-train-ss)
Benchmarking of Inference (Batch Latency + Inference time per Trial) (-benchmark)
Live-Simulation of a single Subject Run (-live_sim)
Configuration Parameters for Number of Epochs, TensorRT Optimizations,... (see main.py --help)

History:
  2021-05-06: Version 0.7 from P. Kessler
  2021-05-10: Parameter mi_ds which specifies which data set should be used,
              e. g. during training - ms (Manfred Strahnen)
  2021-05-11: Ongoing implementation and optimization - ms
"""
import sys

import mne
import torch

from machine_learning.modes import training_cv, benchmarking, live_sim, \
    training_ss
from util.cmd_parser import create_parser, parse_and_check
from util.misc import load_chs_of_model
from data.physionet_dataset import MNE_CHANNELS, excluded_subjects
from data.bcic_dataset import BCIC_CHANNELS, BCIC_excluded_subjects
from config import eeg_config, global_config
from data.physionet_dataset import PHYSIONET
from data.bcic_dataset import BCIC_CONFIG

def single_run(argv=sys.argv[1:]):
    parser = create_parser()
    args = parse_and_check(parser, argv)

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

    # Adjust global parameters which depend on the selected dataset
    if (args.dataset == "PHYS") & (args.ch_names == None) & (args.ch_motorimg == None):
        args.ch_names = MNE_CHANNELS

    if (args.dataset == "PHYS") & (args.excluded == None):
        args.excluded = excluded_subjects

    if (args.dataset == "BCIC") & (args.ch_names == None) & (args.ch_motorimg == None):
        args.ch_names = BCIC_CHANNELS
    if (args.dataset == "BCIC") & (args.excluded == None):
        args.excluded = BCIC_excluded_subjects

    # Dataset dependent EEG config structure re-initialization
    if args.dataset == "PHYS":
        eeg_config.TMIN = PHYSIONET.TMIN
        eeg_config.TMAX = PHYSIONET.TMAX
        eeg_config.TRIAL_SLICES = 1
        eeg_config.SAMPLERATE = PHYSIONET.SAMPLERATE
        eeg_config.SAMPLES=(int) ((PHYSIONET.TMAX - PHYSIONET.TMIN) * PHYSIONET.SAMPLERATE)
    elif args.dataset == "BCIC":
        eeg_config.TMIN = BCIC_CONFIG.TMIN
        eeg_config.TMAX = BCIC_CONFIG.TMAX
        eeg_config.TRIAL_SLICES = 1
        eeg_config.SAMPLERATE = BCIC_CONFIG.SAMPLERATE
        eeg_config.SAMPLES = (int) ((BCIC_CONFIG.TMAX - BCIC_CONFIG.TMIN) * BCIC_CONFIG.SAMPLERATE)

    if args.train:
        global_config.FREQ_FILTER_HIGHPASS=None
        global_config.FREQ_FILTER_LOWPASS=None
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=0.00001
        global_config.FREQ_FILTER_LOWPASS=40.0
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=4.0
        global_config.FREQ_FILTER_LOWPASS=40.0
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=8.0
        global_config.FREQ_FILTER_LOWPASS=40.0
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=12.0
        global_config.FREQ_FILTER_LOWPASS=40.0
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=16.0
        global_config.FREQ_FILTER_LOWPASS=40.0
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=20.0
        global_config.FREQ_FILTER_LOWPASS=40.0
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=24.0
        global_config.FREQ_FILTER_LOWPASS=40.0
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=28.0
        global_config.FREQ_FILTER_LOWPASS=40.0
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=32.0
        global_config.FREQ_FILTER_LOWPASS=40.0
        training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                    equal_trials=(not args.all_trials), early_stop=args.early_stop,
                    excluded=args.excluded, mi_ds=args.dataset)
        global_config.FREQ_FILTER_HIGHPASS=36.0
        global_config.FREQ_FILTER_LOWPASS=40.0
        return training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                           name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                           equal_trials=(not args.all_trials), early_stop=args.early_stop,
                           excluded=args.excluded, mi_ds = args.dataset)
    elif args.train_ss:
        args.ch_names = load_chs_of_model(args.model)
        training_ss(args.model, args.subject, num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                    batch_size=args.bs, tag=args.tag, ch_names=args.ch_names)
    elif args.benchmark:
        args.ch_names = load_chs_of_model(args.model)
        return benchmarking(args.model, name=args.name, n_classes=args.n_classes, device=device,
                            subjects_cs=args.subjects_cs, tensorRT=args.trt, fp16=args.fp16,
                            iters=args.iters, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                            equal_trials=(not args.all_trials), continuous=args.continuous)
    elif args.live_sim:
        args.ch_names = load_chs_of_model(args.model)
        return live_sim(args.model, subject=args.subject, name=args.name, ch_names=args.ch_names,
                        n_classes=args.n_classes, device=device, tag=args.tag)


########################################################################################
if __name__ == '__main__':
    single_run()

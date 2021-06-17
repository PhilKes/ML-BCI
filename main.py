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

    if args.train:
        return training_cv(num_epochs=args.epochs, device=device, n_classes=args.n_classes,
                           name=args.name, batch_size=args.bs, tag=args.tag, ch_names=args.ch_names,
                           equal_trials=(not args.all_trials), early_stop=args.early_stop,
                           excluded=args.excluded, mi_ds=args.dataset, only_fold=args.only_fold)
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

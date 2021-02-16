# EEGNet PyTorch
* EEGNet Implementation in PyTorch
* Training with 5-Fold Cross Validation
* Benchmarking of Inferencing on NVIDIA Jetson Nano

### main.py
Main Script to run Training/Benchmarking of EEGNet

`main.py -train`
* 3Class-Classification Training (Physionet Task 2)
* 5-Fold Crossvalidation
* Saving results and trained model in _./results/training/{DateTime}_

`main.py -benchmark` 
* Inference Benchmarking in batches (default size: 16)
* TensorRT optimization possible with `--trt` flag 
* Saving results (Batch Latency, Inference time per trial) in _./results/benchmark/{DateTime}_

`main.py --help` for all parameters

## main_all_confs.py
* Runs _main.py_ with all possible Configurations in benchmark mode (/w TRT (fp16/32))
* Saves results in parent folder _./results/benchmark/{DateTime}_all_confs_
* Benchmarking different Batch Sizes with `--bs` argument

### config.py
Global Default Configuration/Settings

### EEGNet_model.py
PyTorch Implementation of EEGNet
based on [aliasvishnu/EEGNet](https://github.com/aliasvishnu/EEGNet)

### EEGNet_physionet.py
Main loops for 
* EEGNet Training + Validation on Physionet Dataset and
* Benchmarking of Inferencing over Physionet Dataset with pretrained model

### data_loading.py
* Helper functions to load Physionet Data via MNE Library
* TrialsDataset class for usage with PyTorch Dataloader

### common.py
Main methods for
* Training 
* Validation
* Benchmark

#### utils.py
Miscellaneous Helper methods for logging / saving results / Plots

#### visualize_results.py
* Plots and saves Results from _main_all_confs.py_ Runs as .png
* `--folder` specifies the folder location of the _results.npz_ file

#### python_test.py
Python Playground for testing libraries

## Libraries
[PyTorch](https://pytorch.org/)

[PyCuda](https://documen.tician.de/pycuda/)

[MNE](https://mne.tools/stable/index.html)

[torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

[Matplotlib](https://matplotlib.org/)
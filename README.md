# EEGNet PyTorch
* EEGNet Implementation in PyTorch
* Training with 5-Fold Cross Validation
* Benchmarking of Inferencing on NVIDIA Jetson Nano

### main.py
Main Script to run Training/Benchmarking of EEGNet

`main.py -train`
* n-Class-Classification Training (Physionet Task 2)
* 5-Fold Crossvalidation
* Saving results and trained model in _./results/{DateTime/Name}/training_

`main.py -benchmark` 
* Inference Benchmarking in batches with specified trained model (default size: 16)
* TensorRT optimization possible with `--trt` flag 
* Saving results (Batch Latency, Inference time per trial) in _./results/{model_path}/benchmark_

`main.py --help` for all parameters

### config.py
Global Default Configuration/Settings
* Bandpassfilters
* EEG Trial Interval
* Learning Rate

### models/eegnet.py
PyTorch Implementation of EEGNet
Original Source:
[xiaywang/q-eegnet_torch](https://github.com/xiaywang/q-eegnet_torch/blob/0f467e7f0d9e56d606d8f957773067bc89c2b42c/eegnet.py)

### physionet_machine_learning.py
Main loops for 
* EEGNet Training + Validation on Physionet Dataset using 5-Fold CV
* Benchmarking of Inferencing over Physionet Dataset with pretrained model

### common.py
Main methods for
* Training 
* Validation
* Benchmark

### data_loading.py
* Helper functions to load Physionet Data via MNE Library
* TrialsDataset class for usage with PyTorch Dataloader

## bench_all_confs.py
* Runs _main.py_ with all possible Configurations in benchmark mode (/w TRT (fp16/32))
* Saves results in parent folder _./results/{model_path}/benchmark/_
* Benchmarking with different Batch Sizes with `--bs` argument

#### utils.py
Miscellaneous Helper methods for logging / saving results / Plots

#### visualize_results.py
* Plots and saves Results from _bench_all_confs.py_ Runs as .png
* `--model` specifies the folder location of the _results.npz_ file

#### python_test.py
Python Playground for testing libraries

## Libraries
Use `python3 -m pip install -r requirements.txt` to ensure all necessary libraries are installed

[Numpy](https://numpy.org/)

[PyTorch](https://pytorch.org/)

[PyCuda](https://documen.tician.de/pycuda/)

[MNE](https://mne.tools/stable/index.html)

[torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

[Matplotlib](https://matplotlib.org/)

[Pandas](https://pandas.pydata.org/)
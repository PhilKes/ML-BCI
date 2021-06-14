import numpy as np
from sklearn.metrics import confusion_matrix
import torch  # noqa
from config import eeg_config
from machine_learning.configs_results import get_trained_model_file
from machine_learning.models.eegnet import EEGNet

# Torch to TensorRT for model optimizations
# https://github.com/NVIDIA-AI-IOT/torch2trt
# Comment out if TensorRt is not installed
# if torch.cuda.is_available():
#     import ctypes
# #    from torch2trt import torch2trt
#
#     _cudart = ctypes.CDLL('libcudart.so')


# Source https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
def get_class_accuracies(act_labels, pred_labels):
    conf_mat = confusion_matrix(act_labels, pred_labels)
    cm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    return 100 * cm.diagonal()


def get_trials_per_class(n_class, act_labels):
    class_trials = np.zeros(n_class)
    for i in range(n_class):
        class_trials[i] = np.count_nonzero(act_labels == i, axis=0)
    return class_trials


def get_confusion_matrix(act_labels, pred_labels):
    return confusion_matrix(act_labels, pred_labels)


# Returns EEGNet model optimized with TensorRT (fp16/32)
def get_tensorrt_model(model, batch_size, chs, device, fp16):
    t = torch.randn((batch_size, 1, chs, eeg_config.SAMPLES)).to(device)
    # add_constant() TypeError: https://github.com/NVIDIA-AI-IOT/torch2trt/issues/440
    # TensorRT either with fp16 ("half") or fp32
    if fp16:
        t = t.half()
        model = model.half()
    model = torch2trt(model, [t], max_batch_size=batch_size, fp16_mode=fp16)
    print(f"Optimized EEGNet model with TensorRT (fp{'16' if fp16 else '32'})")
    return model


# Run some example Inferences to warm up the GPU
def gpu_warmup(device, warm_ups, model, batch_size, chs, fp16):
    print("Warming up GPU")
    for u in range(warm_ups):
        with torch.no_grad():
            data = torch.randn((batch_size, 1, chs, eeg_config.SAMPLES)).to(device)
            y = model(data.half() if fp16 else data)


# Return EEGNet model
# pretrained state will be loaded if present
def get_model(n_class, chs, device, model_path=None):
    model = EEGNet(N=n_class, T=eeg_config.SAMPLES, C=chs)
    # model = DoseNet(C=chs, n_class=n_class, T=eeg_config.SAMPLES)
    if model_path is not None:
        model.load_state_dict(torch.load(get_trained_model_file(model_path, n_class)))
    model.to(device)
    return model

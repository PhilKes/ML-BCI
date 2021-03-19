# Add gumpy path
import sys

from config import SAMPLERATE

sys.path.append('gumpy')
from gumpy import signal
import numpy as np


# Source
# https://github.com/rootskar/EEGMotorImagery/blob/master/preprocess.py
# gumpy bci library:
# https://github.com/gumpy-bci/gumpy
def preprocess_mne_data(X):
    X = np.swapaxes(X, 1, 2)
    # print(X[0][1])
    for tr in range(X.shape[0]):
        for ch in range(X.shape[1]):
            ch_data = X[tr][ch]
            X[tr][ch] = preprocess_data(ch_data, sample_rate=SAMPLERATE,
                                        artifact_removal=False, notch=True,
                                        hp_filter=False, bp_filter=False)
    X = np.swapaxes(X, 2, 1)
    return X


def preprocess_data(data, sample_rate=160, ac_freq=60, hp_freq=0.5, bp_low=2, bp_high=60, notch=False,
                    hp_filter=False, bp_filter=False, artifact_removal=False, normalize=False):
    if notch:
        data = notch_filter(data, ac_freq, sample_rate)
    if hp_filter:
        data = highpass_filter(data, hp_freq)
    if bp_filter:
        data = bandpass_filter(data, bp_low, bp_high, sample_rate)
    if normalize:
        data = normalize_data(data, 'mean_std')
    if artifact_removal:
        data = remove_artifacts(data)

    return data


def notch_filter(data, ac_freq, sample_rate):
    w0 = ac_freq / (sample_rate / 2)
    return signal.notch(data, w0)


def highpass_filter(data, hp_freq):
    return signal.butter_highpass(data, hp_freq)


def lowpass_filter(data, lp_freq):
    return signal.butter_lowpass(data, lp_freq)


def bandpass_filter(data, bp_low, bp_high, sample_rate):
    return signal.butter_bandpass(data, bp_low, bp_high, order=5, fs=sample_rate)


def normalize_data(data, strategy):
    return signal.normalize(data, strategy)


def remove_artifacts(data):
    cleaned = signal.artifact_removal(data.reshape((-1, 1)))[0]
    return np.squeeze(cleaned)

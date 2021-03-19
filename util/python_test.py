"""
IGNORE
Python script for miscellaneous testing of libraries
"""
import math

import mne
import numpy as np
import torch

from config import SAMPLERATE, MNE_CHANNELS, EEG_TMAX, EEG_TMIN, SAMPLES
from data_loading import load_n_classes_tasks, remove_n_occurence_of, mne_load_subject_raw, mne_load_subject, \
    load_task_runs

print(F"Torch version:\t{torch.__version__}")
print(F"Cuda available:\t{torch.cuda.is_available()},\t{torch.cuda.device_count()} Devices found. ")
print(F"Current Device:\t{torch.cuda.get_device_name(0)}\t(Device {torch.cuda.current_device()})")

# print("ALL_SUBJECST", ALL_SUBJECTS)
# groups = np.zeros((len(ALL_SUBJECTS)), dtype=np.int)
# splits = 5
# group_size = int(len(ALL_SUBJECTS) / splits)
# for i in range(splits):
#     groups[group_size * i:(group_size * (i + 1))] = i
# unique, counts = np.unique(groups, return_counts=True)
# print("groups", dict(zip(unique, counts)))
# print("groups", groups.shape)
# cv = GroupKFold(n_splits=5)
# cv_split = cv.split(X=ALL_SUBJECTS, groups=groups)
# # for train_index, test_index in cv.split(subjects_data, subjects_labels, groups):
# #     print("TRAIN:", train_index, "TEST:", test_index)
#
# #     print(X_train, X_test, y_train, y_test)
# for i in range(5):
#     print(f"#### {i}th run")
#     train_subjects_idxs, test_subjects_idxs = next(cv_split)
#     train_subjects = [ALL_SUBJECTS[idx] for idx in train_subjects_idxs]
#     test_subjects = [ALL_SUBJECTS[idx] for idx in test_subjects_idxs]
#
#     print(f"Train\n{train_subjects}")
#     print(f"Test \n{test_subjects}")
#     print_subjects_ranges(train_subjects,test_subjects)
#
# print([f"Splits {i}" for i in range(1, 1 * 5 + 1)])
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# def matplot(data, title='', xlabel='', ylabel='', labels=[], save_path=None):
#     plt.figure()
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
#     plt.ylabel(ylabel)
#     plt.grid()
#     if data.ndim == 2:
#         for i in range(len(data)):
#             plt.plot(data[i], label=labels[i] if len(labels) >= i else "")
#             plt.legend()
#     else:
#         plt.plot(data, label=labels[0] if len(labels) > 0 else "")
#     if save_path is not None:
#         fig = plt.gcf()
#         fig.savefig(f"{save_path}/{title}.png")
#     plt.show()
#
# x= np.asarray([0.234,0.2543,0.537,0.264])
# matplot(x,"Test","Epochs","Accuracy")

# path_numpy = "./datasets/numpy"
# data = np.random.random((84, 1281, 64))
# print("Data: ", data.shape)
# try:
#     os.mkdir(path_numpy)
# except OSError as err:
#     pass
# time = datetime.now()


# def save_as_numpy(subject, n_classes, X, y):
#     path = f"{path_numpy}/classes_{n_classes}"
#     try:
#         os.mkdir(path)
#     except OSError as err:
#         pass
#     np.savez(f"{path}/S{subject:03d}.npz", X=X, y=y)
#
#
# def load_from_numpy(subject, n_classes):
#     data = np.load(f"{path_numpy}/classes_{n_classes}/S{subject:03d}.npz")
#     return data['X'], data['y']


mne.set_log_level('WARNING')
# time = datetime.now()
# X, y = mne_load_subject(1, [1, 3, 7, 11])
# print(f"Load: {str(datetime.now() - time)}")
# print("X Shape: ", X.shape)

# save data in file numpy
# data = np.random.random((5, 64))
# labels = [f"Splits {i}" for i in range(5)]
# title = "Test"
# matplot(data, title, "Splits Iteration", "Avg. Accuracy in %",labels, save_path='./')
# data_loaded= np.load(f'./{title}.npy')
# print(data_loaded.shape)
# print(data_loaded)

# print(map_label(x))
# time = datetime.now()
# X, y = load_n_classes_tasks(1, 4)
# print(str(datetime.now()-time))
# print("X:", X, "y:", y.shape)
# rest_indices = np.where(y == 0)
# print(rest_indices)
# X, y = np.delete(X, rest_indices, axis=0), np.delete(y, rest_indices)
# print("X:", X.shape, "y:", y.shape)
# def save_subjects_numpy():
#     n_class= 3
#     all_data = np.zeros((0, 1281, 64),dtype=np.float32)
#     all_labels = np.zeros((0))
#     for subject in ALL_SUBJECTS[:int(len(ALL_SUBJECTS)/2)]:
#         data, labels = load_n_classes_tasks(subject, n_class)
#         all_data, all_labels = np.concatenate((all_data, data)), np.concatenate((all_labels, labels))
#     print("all_data",all_data.shape,"all_labels",all_labels.shape)
#     np.savez('test.npz',X=all_data,y=all_labels)
# def load_subjects_numpy():
#     data = np.load('test.npz')
#     return data['X'],data['y']

# save_subjects_numpy()
# X,y=load_subjects_numpy()

# print("X:",X.shape,"y:",y.shape)
# while True:
#     pass

# data=plot_numpy('./results/2021-02-04 16_34_35-PC/4class-Losses over epochs.npy','Losses over epochs','loss per batch (size = 16)',True)
# print(data[2])
# for i in range(22,43):
#     X,y = load_n_classes_tasks(i,4)
#     print("X",X)
#     print("y",y.shape)
#     print("")
dev = None
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device("cpu")


# scale = lambda x: x * 10000
# ds_train = TrialsDataset(ALL_SUBJECTS, 4, device)
# loader_train = DataLoader(ds_train, 32, sampler=SequentialSampler(ds_train), pin_memory=False, )


# data, labels = next(iter(loader_train))
# print(data.shape, labels.shape)
# print("mean", data.mean(), "std", data.std())

def print_data(loader):
    for data, labels in loader:
        if not torch.any(data.isfinite()):
            print("Not finite data", data)
        if torch.any(data.isnan()):
            print("Nan found ")
        if torch.any(data.isinf()):
            print("Nan found ")


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_square_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        # print("data", data.shape)
        # [Batch_size,1,Samples,Channels]
        # data.shape = [32,1,1281,64]

        channels_sum += torch.mean(data, dim=[0, 1, 2, 3])
        channels_square_sum += torch.mean(data ** 2, dim=[0, 1, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_square_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


# print_data(loader_train)

# mean, std = get_mean_std(loader_train)
# print("mean", mean.shape, "std", std.shape)
# print("mean", mean, "std", std)
def check_bad_data(subjects, n_classes):
    min, max = math.inf, -math.inf
    for idx, i in enumerate(subjects):
        data, labels = load_n_classes_tasks(i, n_classes)
        if np.isneginf(data).any():
            print("negative infinite data")
        if np.isnan(data).any():
            print("Nan found ")
        if np.isinf(data).any():
            print("Not finite data")
        print(f"{i:3d}", " X", data.shape, "y", labels.shape)
        loc_min = data.min()
        loc_max = data.max()
        if (loc_min < min):
            min = loc_min
        if (loc_max > max):
            max = loc_max
    print("Min", min, "Max", max)


# check_bad_data(ALL_SUBJECTS, 4)
# data,labels=mne_load_subject(1,4,{'T0':1,'T2':2})
# data, labels = load_n_classes_tasks(1, 4)
# print(labels, labels.shape)
# print(data.shape)
# x = ALL_SUBJECTS
# batch_size = 16
# m = int(len(ALL_SUBJECTS) / int(math.ceil(len(x) / batch_size)))+1
# test = [x[i:i + m] for i in range(0, len(x), m)]
# # test[-2:] = [test[-2] + test[-1]]
# print(test)
# X, y = load_n_classes_tasks(1, 3, equal_trials=True)
# print(y)
# for i in range(4):
#     print(i, ": ", len(np.where(y == i)[0]))
# scaler = MinMaxScaler(copy=False)
# X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
# print(X.shape)

# print(X[0:21])
#
# X2, y2 = get_data('..\\EEGNet_Tensorflow\\eegnet-based-embedded-bci\\dataset\\files\\', n_classes=3)
# #
# X2 = np.swapaxes(X2, 1, 2)
# #
# X
# , y, rem = remove_n_occurence_of(X, y, 21, 0)
# X2, y2, rem2 = remove_n_occurence_of(X2, y2, 21, 0)
#
# print(X)
# print(X2)
#
# print("X", X.shape, "Y", y.shape)
# print("X2", X2.shape, "Y2", y2.shape)
#
# print("y", y)
# print("y2", y2)
#
# X,X2= normalize_data(X), normalize_data(X2)
# print("X", X.shape, "X2",X2.shape)
# print(X[-1])
# print(X2[0])
# X = X.flatten()
# X2 = X2.flatten()
# print(np.all(np.in1d(X, X2)))
# for i,n in enumerate(X):
#     if n not in X2:
#         print(f"{n} (idx {i}) is not present in X2")
#         pass
#
tdelta = EEG_TMAX - EEG_TMIN


def crop_time_and_label(raw, time, ch_names=MNE_CHANNELS):
    if (time - tdelta) < 0:
        raise Exception(f"Cant load {tdelta}s before timepoint={time}s")
    raw1 = raw.copy()
    raw1.pick_channels(ch_names)
    raw1.crop(time - tdelta, time)
    data, times = raw1[:, :]
    return data, times, raw1.annotations


def get_data_from_raw(raw, ch_names=MNE_CHANNELS):
    raw1 = raw.copy()
    raw1.pick_channels(ch_names)
    data, times = raw1[:, :]
    return data


def get_label_at_idx(times, annot, sample):
    now_time = times[sample]
    if sample < SAMPLES:
        return None,now_time
    middle_sample_of_window = int(sample - (SAMPLES / 2))
    time = times[middle_sample_of_window]
    onsets = annot.onset
    # boolean_array = np.logical_and(onsets >= time, onsets <= time + tdelta)
    # find index where time would be inserted
    # -> index of label is sorted_idx-1
    sorted_idx = np.searchsorted(onsets, [time])[0]
    # Determine if majority of samples lies in
    # get label of sample_idx in the middle of the window
    label = annot.description[sorted_idx - 1]
    return label, now_time


def get_label_at_time(raw, times, time):
    idx = raw.time_as_index(time)
    return get_label_at_idx(times, raw.annotations, idx)



# is constantly increased -> simulate "live"
# current_sample = 0
raw = mne_load_subject_raw(1, [4])
max_sample = raw.n_times
# X, times, annot = crop_time_and_label(raw, 8)
X = get_data_from_raw(raw)
last_label = None
for sample in range(max_sample):
    # get_label_at_idx( times, annot, 10)
    label, now = get_label_at_idx(raw.times, raw.annotations, sample)
    if last_label != label:
        print(f"Label from {now} is: {label}")
    last_label = label

# raw.plot(n_channels=1)

# X, y = mne_load_subject(1, 2)
# print(X.shape)
# X = np.swapaxes(X, 1, 2)
# print(X[0][1])
# for tr in range(X.shape[0]):
#     for ch in range(X.shape[1]):
#         ch_data = X[tr][ch]
#         X[tr][ch] = preprocess_data(ch_data,sample_rate=SAMPLERATE,
#                                     artifact_removal=True,notch=True,
#                                     bp_filter=True)
# X = np.swapaxes(X, 2, 1)
# print("PREPROCESSED.-------------------------------------------------")
# print(X[0][1])
# print(X.shape)

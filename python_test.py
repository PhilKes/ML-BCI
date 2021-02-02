import os
from datetime import datetime

import mne
import torch
import numpy as np
from sklearn.model_selection import GroupKFold

from common import ALL_SUBJECTS, print_subjects_ranges, mne_load_subject

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

path_numpy = "./datasets/numpy"
data = np.random.random((84, 1281, 64))
print("Data: ", data.shape)
try:
    os.mkdir(path_numpy)
except OSError as err:
    pass
time = datetime.now()

def save_as_numpy(subject, n_classes, X, y):
    path = f"{path_numpy}/classes_{n_classes}"
    try:
        os.mkdir(path)
    except OSError as err:
        pass
    np.savez(f"{path}/S{subject:03d}.npz", X=X, y=y)


def load_from_numpy(subject, n_classes):
    data = np.load(f"{path_numpy}/classes_{n_classes}/S{subject:03d}.npz")
    return data['X'], data['y']

print("MNE")
mne.set_log_level('WARNING')
# time = datetime.now()
# X, y = mne_load_subject(1, [1, 3, 7, 11])
# print(f"Load: {str(datetime.now() - time)}")
# print("X Shape: ", X.shape)
#
# print("NUMPY")
# time = datetime.now()
# save_as_numpy(1, 3, X,y)
# print(f"Save: {str(datetime.now() - time)}")
# time= datetime.now()
# X,y= load_from_numpy(1,3)
# print(f"Load: {str(datetime.now() - time)}")

# for i in ALL_SUBJECTS:
#     X,y=mne_load_subject(i,[1,3,7,11])
#     print(f"Subject {i}",X.shape,y.shape)

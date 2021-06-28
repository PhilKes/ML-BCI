"""
IGNORE
Python script for miscellaneous testing of libraries
"""
import math

import mne
import numpy as np
import torch

from config import ROOT
from data.datasets.phys.phys_data_loading import PHYS_DataLoader

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
#     np.savez('util.npz',X=all_data,y=all_labels)
# def load_subjects_numpy():
#     data = np.load('util.npz')
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
        data, labels = PHYS_DataLoader.load_n_classes_tasks(i, n_classes)
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
# util = [x[i:i + m] for i in range(0, len(x), m)]
# # util[-2:] = [util[-2] + util[-1]]
# print(util)
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


# is constantly increased -> simulate "live"
# current_sample = 0
# raw = mne_load_subject_raw(1, [4])
# max_sample = raw.n_times
# # X, times, annot = crop_time_and_label(raw, 8)
# X = get_data_from_raw(raw)
# last_label = None
# for now_sample in range(max_sample):
#     # get_label_at_idx( times, annot, 10)
#     label, now_time = get_label_at_idx(raw.times, raw.annotations, now_sample)
#     if last_label != label:
#         print(f"Label from {now_time} is: {label}")
#     last_label = label

# raw=mne_load_subject_raw(1,[4],fmin=4,fmax=60)
# raw.plot_psd(area_mode='range', tmax=10.0, average=False)

# physionet_live_sim("./results/2_3_class_defaults",n_classes=[3])

# preloaded_data, preloaded_labels = load_subjects_data(ALL_SUBJECTS, 3)
# preloaded_data, preloaded_labels = load_subjects_data(ALL_SUBJECTS, 4)

# physionet_training_ss(1, "./results/excluded_test", device=device,n_classes=[2])


#
# def save_pd(arr,i):
#     with open(os.path.join(f"./results/util", f'batch_training_results{i}.txt'),
#               'w') as outfile:
#         df = pandas.DataFrame(data=arr, columns=['2clAcc','2clOF','3clAcc','3clOF'])
#         df.to_string(outfile)
#
# arr2= np.asarray([[['r1cl2Acc','r1cl3Acc'],['r1cl2Of','r1cl3OF']],
#                  [['r2cl2Acc','r2cl3Acc'],['r2cl2Of','r2cl3OF']]])
# #print(arr)
# arr= arr2.copy().reshape((2,2*2),order='C')
# print(arr)
# save_pd(arr,1)
# arr= arr2.copy().reshape((2,2*2),order='F')
# print(arr)
# save_pd(arr,2)
# arr= arr2.copy().reshape((2,2*2),order='A')
# print(arr)
# save_pd(arr,3)

# set_eeg_times(0, 2)
# live_sim("./results/2_3_class_params/tmax/conf_tmax_2", n_classes=[3])
# set_eeg_times(0, 3)
# live_sim("./results/2_3_class_params/tmax/conf_tmax_3", n_classes=[3])
# set_eeg_times(0, 4)
# live_sim("./results/2_3_class_params/tmax/conf_tmax_4", n_classes=[3])
# set_eeg_times(-1, 5)
# live_sim("./results/2_3_class_params/tmax/conf_tmin_-1_tmax_5", n_classes=[3])
# set_eeg_times(0, 2)

# x1=get_trials_size(2)
# x2=get_trials_size(3)
# x3=get_trials_size(3,ignored_runs=[12,4])
# x4=get_trials_size(4)
# print(x1,x2,x3,x4)
# path=f"{results_folder}/2class_excluded_params/excluded/conf_excl_42{training_results_folder}{training_ss_results_folder}"
# for x in os.listdir(path):
#     print(os.path.join(path, x))
#

# raw= mne_load_subject_raw(1,4)
# raw.plot(n_channels=1)
# raw.plot_sensors()


# plot_live_sim_subject_run()

# # matplot_legend(labels=['Batch Size 8', 'Batch Size 16', 'Batch Size 32'], font_size=28.0, hor=True,
# #               save_path='./results/plots_training3', title='bs_legend')
#
# # plots_training4:
# defaults = np.asarray([82.131519, 72.637944, 65.272109])
#
# # Trials Slicing
# cl2_accs = np.asarray([71.76, 58.78, 60.651927])
# cl3_accs = np.asarray([56.15, 47.49, 40.733182])
# cl4_accs = np.asarray([41.802721, 35.10, 32.285998])
# # plot_accuracies(cl2_accs, cl3_accs, cl4_accs,
# #                 'Trials Slicing Accuracies',
# #                 ['2', '4', '8'],
# #                 './results/plots_training4/slicing',
# #                 xlabel='k Slices',
# #                 defaults=defaults
# #                 )
#
# # Channel Selection
# cl2_accs = np.asarray([78.752834, 79.433107, 79.297052, 80.589569])
# cl3_accs = np.asarray([67.951625, 67.089947, 68.465608, 71.156463])
# cl4_accs = np.asarray([56.893424, 56.734694, 58.015873, 61.678005])
# # plot_accuracies(cl2_accs, cl3_accs, cl4_accs,
# #                 'EEG 16-Channel Selections Accuracies',
# #                 ['chs_16', 'chs_16_2', 'chs_16_openbci', 'chs_16_bs'],
# #                 './results/plots_training4/chs',
# #                 defaults=defaults,
# #                 )
#
# # Time Window
# cl2_accs = np.asarray([80.793651, 82.244898, 81.655329, 82.539683, 88.548753])
# cl3_accs = np.asarray([73.091459, 74.270597, 75.177627, 70.506425, 80.982615])
# cl4_accs = np.asarray([62.335601, 65.079365, 65.272109, 65.034014, 71.031746])
# # plot_accuracies(cl2_accs, cl3_accs, cl4_accs,
# #                 'Trial Time Window Accuracies',
# #                 ['[0;1]', '[0;3]', '[-0.5;3]', '[0;4]', '[-1;5]'],
# #                 './results/plots_training4/tmin_tmax',
# #                 xlabel='Time Window in sec.',
# #                 defaults=defaults
# #                 )
#
# # plot_live_sim_subject_run(subject=1, n_class=3, ch_names=[i for i in MNE_CHANNELS if i not in MOTORIMG_CHANNELS_16])
# # plot_live_sim_subject_run(subject=1, n_class=3, ch_names=MOTORIMG_CHANNELS_16)
#
# plot_confusion_matrices("../results/plots_training4/defaults/conf_defaults/training/")
#
# #load_and_plot_training("./results/plots_training4/defaults/conf_defaults/training/")
#
# x=np.asarray(
#     [59.333043,
# 60.312492,
# 60.085578,
# 60.750915,
# 60.071384,
# 64.281474,
# 60.796708,
# 61.757666,
# 56.666946,
# 54.434232,
# 54.468478,
# 53.507223,
# 53.016233,
# 51.186240,
# 53.104565,
# 51.784979,
# 52.177341,
# 51.263150,
# 53.391648,
# 52.123931,])
# fbs=4
# tmins=5
# x=x.reshape((x.shape[0],1))
# x=subtract_first_config_accs(x,fbs)
# x = x.reshape(fbs - 1, tmins, order='F')
# print(x)
# import pandas as pd
# names=['t1','t2','t3','t4','t5']
# columns=['f1','f2','f3']
# df = pd.DataFrame(data=x, index=columns, columns=names)
# print(df)
# data = np.asarray([
#
#     [90.977444,
#     93.984962,
#     49.624060,
#     63.909774],
#
#     [79.699248,
#     73.684211,
#     54.135338,
#     57.894737],
#
#     [69.172932,
#     73.684211,
#     54.135338,
#     61.654135],
#
#     [51.879699,
#     48.120301,
#     48.872180,
#     55.639098],
#
#     [44.360902,
#     45.864662,
#     55.639098,
#     52.631579]
# ])
# labels=['all','f1','f2','f3']
# x_values=['-','0.0-2.0s','0.5-2.5s','1.0-3.0s','1.5-3.5s','2.0-4.0s']
# matplot(data.T, title='Frequency Band Accuracies for 2s Time Slices', xlabel='2s Time Slice Interval', ylabel='Accuracy in %', labels=labels, bar_plot=False,
#             x_values=x_values, ticks=None,min_x=0,marker='o',fig_size=(8.0,6.0))

# file='./util.npz'
# np.savez(file,test_string="Test")
# test_string= np.load(file)['test_string']
# print(test_string)


print(ROOT)

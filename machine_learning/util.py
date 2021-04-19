import numpy as np
from sklearn.metrics import confusion_matrix


# Source https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
def get_class_accuracies(act_labels, pred_labels):
    conf_mat = confusion_matrix(act_labels, pred_labels)
    cm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    return 100 * cm.diagonal()


# Source https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
def get_trials_per_class(n_class, act_labels):
    class_trials = np.zeros(n_class)
    for i in range(n_class):
        class_trials[i] = np.count_nonzero(act_labels == i, axis=0)
    return class_trials


def get_confusion_matrix(act_labels, pred_labels):
    return confusion_matrix(act_labels, pred_labels)

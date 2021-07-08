import argparse
import os

import scipy
from scipy import signal
from tqdm import tqdm
import numpy as np
from data.datasets.lsmr21.lsmr21_data_loading import LSMR21DataLoader, LSMRSubjectRun, LSMRTrialData
from util import misc


def subject_run_to_numpy(sr: LSMRSubjectRun, path, ds_factor=4):
    data = sr.data
    # trial_info = (label, trial_category, artifact, triallength)
    trial_info = np.zeros((data.shape[0], 4))
    for i in range(data.shape[0]):
        samples = int(data[i].shape[-1] // ds_factor)
        data[i] = signal.resample(data[i], samples, axis=1)
        trialdata = sr.trialdata[i]
        trial_info[i] = np.asarray(
            [trialdata.targetnumber, get_trial_category(trialdata), trialdata.artifact, trialdata.triallength],
            dtype=np.float16)
    np.savez(
        path,
        data=data,
        trial_info=trial_info,
        subject=sr.subject,
    )


#
def get_trial_category(trialdata: LSMRTrialData) -> int:
    if trialdata.result == 1:
        return 1
    if trialdata.forcedresult == 1:
        return 2
    return 0


from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Script to convert original "Human EEG Dataset for Brain-Computer Interface and Meditation" Matlab Files to Numpy Files')
    parser.add_argument('--ds_factor', type=float,default=4,
                        help=f"Downsample-Factor (1000Hz / ds_factor = new Samplerate)")
    parser.add_argument('--origin_path', type=str, default='/opt/datasets/LSMR21/matlab',
                        help=f"Path to Folder containing Matlab Dataset files")
    parser.add_argument('--dest_path', type=str, default=None,
                        help=f"Path to Folder for the converted numpy files")

    args = parser.parse_args()
    if not os.path.exists(args.origin_path):
        parser.error(f"Origin path '{args.origin_path}' does not exist!")

    if args.dest_path is None:
        path = Path(args.origin_path)
        args.dest_path = os.path.join(path.parent.absolute(), 'numpy')
    if not os.path.exists(args.dest_path):
        misc.makedir(args.dest_path)
    matlab_files = sorted([file for file in os.listdir(args.origin_path) if file.endswith('.mat')])
    for i, files in enumerate(tqdm(matlab_files)):
        matlab_data = misc.load_matlab(os.path.join(args.origin_path, files))
        sr = LSMRSubjectRun(i, matlab_data)
        file_name, file_ext = os.path.splitext(files)
        subject_run_to_numpy(sr, os.path.join(args.dest_path, file_name), ds_factor=args.ds_factor)

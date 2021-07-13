"""
Script to download and convert the LSMR21 Dataset into numpy
if -download argument is present, downloads Dataset's Matlab Files into local directory
then converts all Matlab data into numpy (.npz) files with minimal neccessary data
"""
import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
from scipy import signal
from tqdm import tqdm

from config import datasets_folder
from data.datasets.lsmr21.lmsr21_matlab import LSMRTrialData
from data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from data.datasets.lsmr21.lsmr21_data_loading import LSMRSubjectRun
from util import misc


def subject_run_to_numpy(sr: LSMRSubjectRun, path, ds_factor=4):
    """
    Converts Matlab Data Structure of 1 Subject Run into numpy file (.npz)
    :param sr: Matlab Data
    :param path: Destination path for .npz file
    :param ds_factor: downsampling Factor (1000Hz original Samplerate)
    """
    data = sr.data
    # trial_info = (label, tasknr, trial_category, artifact, triallength)
    trial_info = np.zeros((data.shape[0], 5))
    for i in range(data.shape[0]):
        samples = int(data[i].shape[-1] // ds_factor)
        data[i] = signal.resample(data[i], samples, axis=1)
        trialdata = sr.trialdata[i]
        trial_info[i] = np.asarray(
            [trialdata.targetnumber, trialdata.tasknumber, get_trial_category(trialdata),
             trialdata.artifact, trialdata.triallength],
            dtype=np.float16)
    np.savez(
        path,
        data=data,
        trial_info=trial_info,
        subject=sr.subject,
    )


def get_trial_category(trialdata: LSMRTrialData) -> int:
    """
    Categorize Trials by result/forcedresult
    """
    if trialdata.result == 1:
        return 1
    if trialdata.forcedresult == 1:
        return 2
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=f"Script to download/convert original '{LSMR21.name}'")
    parser.add_argument('--ds_factor', type=float, default=4,
                        help=f"Downsample-Factor (1000Hz / ds_factor = new Samplerate)")
    parser.add_argument('--origin_path', type=str, default=f"{datasets_folder}/{LSMR21.short_name}/matlab/",
                        help=f"Path to Folder containing Matlab Dataset files")
    parser.add_argument('--dest_path', type=str, default=None,
                        help=f"Path to Folder for the converted numpy files "
                             f"(default: subdir 'numpy' in parentdir of --origin_path)")
    parser.add_argument('-download',action='store_true', required=False,
                        help="If present, downloads all Matlab Files of the Dataset from Figshare.com into"
                             " --origin_path before converting to numpy")
    args = parser.parse_args()
    if args.download & (args.origin_path is None):
        parser.error("You need to specify a destination path for the Matlab Files (--origin_path)!")

    # Download all Matlab Files of Dataset from Figshare
    if args.download:
        if not os.path.exists(args.origin_path):
            misc.makedir(args.origin_path)
        print(f"Downloading all {LSMR21.short_name} Matlab Files from Figshare.com into '{args.origin_path}'")
        with urllib.request.urlopen("https://api.figshare.com/v2/articles/13123148/files") as url:
            files_list = json.loads(url.read().decode())
            for file in tqdm(files_list):
                dl_path = os.path.join(args.origin_path, file['name'])
                # Skips Files that are already present in origin_path
                if os.path.exists(dl_path):
                    continue
                # Download and store file in origin_path
                urllib.request.urlretrieve(file['download_url'], dl_path)
        print(f"Finished downloading all Matlab Files into '{args.origin_path}'")

    if not os.path.exists(args.origin_path):
        parser.error(f"Origin path '{args.origin_path}' does not exist!")

    if args.dest_path is None:
        path = Path(args.origin_path)
        args.dest_path = os.path.join(path.parent.absolute(), 'numpy')

    if not os.path.exists(args.dest_path):
        misc.makedir(args.dest_path)
    # Get Matlab Files in origin_path
    matlab_files = sorted([f for f in os.listdir(args.origin_path) if f.endswith('.mat')])
    print(f"Converting all {len(matlab_files)} .mat Files from '{args.origin_path}'"
          f" to minimal .npz Files in '{args.dest_path}'")
    for file in tqdm(matlab_files):
        mat_file_name, mat_file_ext = os.path.splitext(file)
        npz_file=os.path.join(args.dest_path, f"{mat_file_name}.npz")
        if os.path.exists(npz_file):
            continue
        # Load Matlab Data of 1 Subject Run
        matlab_data = misc.load_matlab(os.path.join(args.origin_path, file))
        # Get Subject Nr. from Filename
        subject = int(re.findall(r'S(.+)_Session', file)[0])
        sr = LSMRSubjectRun(subject, matlab_data)
        # Convert Subject Run to necessary numpy data and store in .npz file
        subject_run_to_numpy(sr, npz_file, ds_factor=args.ds_factor)

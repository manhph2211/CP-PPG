import torch
import sys
sys.path.append(".")
import glob
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
from src.utils.utils import get_config, plot_wavform, get_peaks_info, write_json, plot_subject_distribution, read_json, write_json, standardize
from src.utils.preprocess import read_processed_signal, get_scaler, normalize, extract_window_segments, cycle_helper, read_indices
from src.utils.classification import classify
from src.utils.enrichment import PPGTransform
from configs.seed import *
from scipy import signal
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk


class DataHandler:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.window = cfgs['data']['window'] 
        self.stride = cfgs['data']['stride'] 
        self.ratio = cfgs['data']['train_size']
        self.transform = PPGTransform()
        self.subject_to_segments = defaultdict(list)
        self.ref_bank = []
        self.max_len = -1
        self.min_len = 128
        self.cycle_lens = []
        self.test_cycle_lens = []
        self.data_snapshot = None
        
    def custom_data_hanlder(self, version="8s"):
        data = read_json(os.path.join("assets/presmooth/", f"Max1Not2L_{version}_75overlap_NOECG.json"))      

        custom_data = defaultdict(list)

        for key, value in tqdm(data.items()):
            subject = key.split(".")[0][-4:]
            if subject in ['1024','1026','1001','1020']:
                continue # very few in WF-PPG
            in_windows = value["in_windows"]
            ref_windows = value["ref_windows"]
            ref_indices = value["ref_indices"]

            in_windows = np.array(in_windows)
            ref_windows = np.array(ref_windows)
            for i in range(len(in_windows)):
                in_window = standardize(in_windows[i])
                ref_window = standardize(ref_windows[i])
                in_window = in_window.reshape(1, -1)
                ref_window = ref_window.reshape(1, -1)

                custom_data[subject].append([in_window.tolist(), ref_window.tolist()])
            
        assert len(glob.glob(f"assets/refs/*.npy")) <= len(custom_data)
        write_json(custom_data, f"assets/data.json")
        print("DONE PREPROCESS!")

    def train_val_test_split(self, version="8s"):

        self.data = read_json("assets/data.json")
        subjects = list(self.data.keys())
        test_subjects =  ["1012","1003","1010","1007","1008","1028"] 
        
        train_subjects = [subject for subject in subjects if subject not in test_subjects]

        augmented_train_val_data = {subject: read_json("assets/data.json")[subject] for subject in train_subjects}

        train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.1, shuffle=False)

        train_data = {subject: augmented_train_val_data[subject] for subject in train_subjects}
        val_data = {subject: augmented_train_val_data[subject] for subject in val_subjects}
        test_data = {subject: self.data[subject] for subject in test_subjects}

        write_json(train_data, f"assets/train{version}.json")
        write_json(val_data, f"assets/val{version}.json")
        write_json(test_data, f"assets/test{version}.json")

    def kfold_split(self, random_state=42, version="8s"):
        custom_data = read_json("assets/data.json")
        augmentation = "aug" if self.cfgs['data']['enrich'] else "noaug"

        custom_test_subjects = [[1023, 1002, 1013, 1031],[1003,1008,1005,1030],[1012,1022,1010,1007],[1018,1021,1000,1009,1028],[1017,1011, 1004, 1019,1029]]

        custom_test_subjects = [[str(item) for item in inner_list] for inner_list in custom_test_subjects]

        for fold_index, test_subjects in enumerate(custom_test_subjects):
            train_subjects = [subject for subject in custom_data if subject not in test_subjects]

            print(f"Test Subjects in Fold {fold_index + 1}: ", test_subjects)
            print(f"Train Val Subjects in Fold {fold_index + 1}: ", train_subjects)

            test_data = {subject: custom_data[subject] for subject in test_subjects}
            write_json(test_data, f"assets/{augmentation}/{version}_test_fold_{fold_index + 1}.json")

            augmented_train_val_data = {subject: read_json("assets/data.json")[subject] for subject in train_subjects}

            for subject in train_subjects:

                if self.cfgs['data']['enrich']:
                    if subject in ['1023', '1022', '1012', '1003', '1002', '1008', '1018']:

                        for in_window, ref_window in tqdm(custom_data[subject]):
                            new_in_window = np.array(in_window).reshape(-1).copy()
                            new_ref_window = ref_window.copy()
                            for k in range(3):
                                augmented_train_val_data[subject].append([self.transform.convert(new_in_window).reshape(1, -1).tolist(), new_ref_window])

            kf_inner = KFold(n_splits=4, shuffle=True, random_state=random_state)
            inner_fold_iter = kf_inner.split(train_subjects)
            train_inner_index, val_index = next(inner_fold_iter)  
            train_inner_subjects = [train_subjects[i] for i in train_inner_index]
            val_subjects = [train_subjects[i] for i in val_index]
            print(f"Val Subjects in Fold {fold_index + 1}: ", val_subjects)
            print(val_subjects)

            train_data = {subject: augmented_train_val_data[subject] for subject in train_inner_subjects}
            val_data = {subject: augmented_train_val_data[subject] for subject in val_subjects}

            write_json(train_data, f"assets/{augmentation}/{version}_train_fold_{fold_index + 1}.json")
            write_json(val_data, f"assets/{augmentation}/{version}_val_fold_{fold_index + 1}.json")

    def __call__(self, version="8s"):
        self.custom_data_hanlder(version=version)
        self.train_val_test_split(version=version)


if __name__=='__main__':
    cfgs = get_config()
    data_handler = DataHandler(cfgs)
    for version in ["3s","5s","8s"]:
        print(f"############################## START PROCESS {version}-WINDOWS ##############################")
        data_handler(version)

import torch
import sys
sys.path.append(".")
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from src.utils.utils import get_config, plot_wavform, read_json
from configs.seed import *


class PPGDataset(Dataset):
    def __init__(self, cfgs, data_path):
        self.data = []
        self.segments = []
        for key, values in read_json(data_path).items():
            self.segments.extend(values)
            for value in values:
                self.data.append([key, value])

        self.segments = np.array(self.segments)

        self.mode = data_path.split("/")[-1][:-5]
        print(f"PROCESSING {self.mode} DATASET ... ")
        self.cfgs = cfgs
        self.mask = (self.segments[:,0,:,:] != 0).astype(float)
        self.src_segments, self.ref_segments = self.segments[:,0,:,:], self.segments[:,1,:,:]

        self.ref_files = glob.glob(f"assets/refs/*.npy")
        self.ref_batch = []
        self.look_up = {}
        for ref_file in self.ref_files:
            subject_name = ref_file.split("/")[-1][:-4]
            ref_sample = np.load(ref_file).reshape(-1)
            self.ref_batch.extend(ref_sample)
            self.look_up[subject_name] = ref_sample
        
    def __len__(self):
        return len(self.src_segments)
    
    def __getitem__(self, idx):
        seg_in = self.src_segments[idx]
        subject_name = self.data[idx][0]

        if self.cfgs['data']['film']:
            seg_film = np.array(self.look_up[subject_name])
            # seg_film = self.ref_bank
            if self.cfgs['model']['film']['latent']:
                seg_film = np.array([seg_film])
        else:
            seg_film = self.ref_segments[idx]
        
        seg_ref = self.ref_segments[idx]

        return torch.FloatTensor(seg_in), torch.FloatTensor(seg_ref), torch.FloatTensor(seg_film), torch.FloatTensor(self.mask[idx])


def get_loader(cfgs, version="8s", fold=None):
    print(f"**************** USING {version} DATASET ****************")
    if fold is None:
        fold = cfgs['data']['fold']
        
    augmentation = "aug" if cfgs['data']['enrich'] else "noaug"

    if int(fold):
        print("USING 5 FOLD CV")
        train_dataset = PPGDataset(cfgs, data_path=f"assets/{augmentation}/{version}_train_fold_{fold}.json")
        val_dataset = PPGDataset(cfgs, data_path=f"assets/{augmentation}/{version}_val_fold_{fold}.json")
        test_dataset = PPGDataset(cfgs, data_path=f"assets/{augmentation}/{version}_test_fold_{fold}.json")
    else:
        print("USING NORMAL TRAIN VAL TEST SPLIT")
        train_dataset = PPGDataset(cfgs, data_path=f"assets/train.json")
        val_dataset = PPGDataset(cfgs, data_path=f"assets/val.json")
        test_dataset = PPGDataset(cfgs, data_path=f"assets/test.json")


    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = cfgs['data']['batch_size'],
        num_workers = cfgs['data']['num_workers'],
        shuffle=True
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = cfgs['data']['batch_size'],
        num_workers = cfgs['data']['num_workers'],
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = cfgs['data']['batch_size'],
        num_workers = cfgs['data']['num_workers'],
    )   
    print("DATA SPLIT: ", len(train_loader),len(val_loader),len(test_loader))

    print("DONE LOADING DATA !")
    return train_loader, val_loader, test_loader


if __name__=='__main__':
    cfgs = get_config()
 
    train_set, val_set, test_set = get_loader(cfgs)
    print(len(train_set),len(val_set),len(test_set))
    src_signal, ref_signal, _, _ = next(iter(iter(test_set)))
    src_signal, ref_signal = src_signal.numpy(), ref_signal.numpy()
    plot_wavform(src_signal.reshape(-1)[:1200], ref_signal.reshape(-1)[:1200])
    plot_wavform(src_signal.reshape(-1)[:1200], ref_signal.reshape(-1)[:1200])
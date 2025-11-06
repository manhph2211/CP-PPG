import torch
import numpy as np
import pandas as pd
from tqdm import tqdm 
from scipy import signal
import csv
import warnings
warnings.simplefilter("ignore", UserWarning)
import sys
sys.path.append(".")
from torch.utils.data import Dataset, DataLoader, RandomSampler
from src.metrics.losses import CustomLoss
from src.models.cpppg import Generator
from src.utils.utils import *
from src.utils.preprocess import *
from src.metrics.metrics import SignalComparison
from src.dataloader.dataset import PPGDataset
from configs.seed import *


class Inference:
    def __init__(self, cfgs, checkpoint=None):
        self.cfgs = cfgs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Generator(get_config()).to(self.device)
        print(self.model)
        self.criterion = CustomLoss(get_config()).to(self.device)
        self.model.eval()
        self.get_quantitative_results = SignalComparison(cfgs)
        if checkpoint is not None:
            self.ckpt = checkpoint
        else:
            self.ckpt = self.cfgs['train']['ckpt']
        
        self.load_weights()

        self.ref_files = glob.glob(f"assets/refs/*.npy")
        self.ref_batch = []
        self.look_up = {}
        for ref_file in self.ref_files:
            subject_name = ref_file.split("/")[-1][:-4]
            ref_sample = np.load(ref_file).reshape(-1)
            self.ref_batch.extend(ref_sample)
            self.look_up[subject_name] = ref_sample

    def load_weights(self):
        try:
            print(f"USING {self.device} ... ")
            print(f"LOADING CHECKPOINTS FROM {self.ckpt} ... ")
            self.model.load_state_dict(torch.load(self.ckpt, map_location=self.device))
            print(f"SUCCESSFULLY LOAD TRAINED MODELS: {self.ckpt} !")
        except:
            print('LACK OF NECESSARY CHECKPOINTS!!!')
    
    def infer(self, in_segments=None, batch_size = 1):
        print(f"Using {self.cfgs['train']['ckpt']}")
        if in_segments is None:
            return None
        in_segments = np.array(in_segments)
        batch_size = in_segments.shape[0]
        num_batches = in_segments.shape[0] // batch_size

        output_batches = []
        for i in range(num_batches):
            batch_in = in_segments[i*batch_size : (i+1)*batch_size, :]
            with torch.no_grad():

                batch_in = torch.FloatTensor(batch_in)
                seg_film = torch.FloatTensor(batch_in) 

                output_batch_tensor = self.model(batch_in.to(self.device), seg_film.to(self.device))  
                output_batches.append(output_batch_tensor.cpu().detach().numpy())

        clean_output = np.vstack(output_batches)
        clean_output = clean_output.reshape(batch_size, -1)
        return clean_output
    
    def infer_loader(self, test_loader):
        init_result = []
        final_result = []
        output_batches = []
        with torch.no_grad():
            for src_signal, ref_signal, src_pressure, mask in tqdm(test_loader):
                src_signal = src_signal.to(self.device)
                src_pressure = src_pressure.to(self.device)    
                ref_signal = ref_signal.to(self.device)
                mask = mask.to(self.device)
                output_batch_tensor = self.model(src_signal, src_pressure, mask)  
                
                init_result.append(self.get_quantitative_results.compare(src_signal, ref_signal))
                final_result.append(self.get_quantitative_results.compare(output_batch_tensor.to(self.device), ref_signal))
                output_batches.append(output_batch_tensor.cpu().detach().numpy())

        print("DONE INFERRING!")

        return np.mean(np.array(final_result), axis=0), np.mean(np.array(init_result), axis=0)

    
if __name__ == "__main__":
    cfgs = get_config()
    test_dataset = PPGDataset(cfgs, data_path="assets/test.json")

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = cfgs['data']['batch_size'],
        num_workers = cfgs['data']['num_workers'],
    )  
    
    infer_tool = Inference(cfgs)
    rec, raw = infer_tool.infer_loader(test_loader)
    print(rec)
    print(raw)
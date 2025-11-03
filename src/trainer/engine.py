from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
import sys
sys.path.append(".")
from metrics.losses import CustomLoss
from src.dataloader.dataset import get_loader
from src.models.cpppg import Generator
from src.utils.utils import plot_result, depadding
import random
from configs.seed import *


class Trainer:
    def __init__(self, tracking, cfgs):
        self.tracking = tracking
        self.cfgs = cfgs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_value = self.cfgs['train']['clip_value']
        self.epoch_n = self.cfgs['train']['epoch_n']
        self.BEST_LOSS = np.inf
        self.ckpt = self.cfgs['train']['ckpt']

        self.model = Generator(cfgs).to(self.device)        
        self.criterion = CustomLoss(self.cfgs).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), betas=(0.9, 0.999), weight_decay=0.005, lr=self.cfgs['train']['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=8, verbose=True)
        self.train_loader, self.val_loader, self.test_loader = get_loader(self.cfgs)
        # self.load_weights()

    def load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.ckpt, map_location=self.device))
            print("SUCCESSFULLY LOAD TRAINED MODELS !")
        except:
            print('FIRST TRAINING')

    def train_epoch(self):
        self.model.train()
        train_loss_epoch = 0
        train_cosin_loss_epoch = 0
        train_peak_to_peak_loss = 0
        train_mse_loss = 0

        for src_signal, ref_signal, seg_film, mask in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            
            src_signal = src_signal.to(self.device)
            seg_film = seg_film.to(self.device)
            out = self.model(src_signal, seg_film, mask.to(self.device))

            ref_signal = ref_signal.to(self.device)
            total_loss, cosin_loss, peak_to_peak_loss, mse_loss = self.criterion(out, ref_signal)
            train_loss_epoch += total_loss.item()
            train_cosin_loss_epoch += cosin_loss.item()
            train_peak_to_peak_loss += peak_to_peak_loss.item()
            train_mse_loss += mse_loss.item()
            total_loss.backward()
            # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)
            self.optimizer.step()

        return train_loss_epoch/ len(self.train_loader), train_cosin_loss_epoch / len(self.train_loader), train_peak_to_peak_loss/ len(self.train_loader), train_mse_loss / len(self.train_loader)

    def val_epoch(self):
        self.model.eval()
        val_loss_epoch = 0
        val_cosin_loss_epoch = 0
        val_peak_to_peak_loss = 0
        val_mse_loss = 0
        with torch.no_grad():
            for src_signal, ref_signal, seg_film, mask in tqdm(self.val_loader):
                src_signal = src_signal.to(self.device)
                seg_film = seg_film.to(self.device)
                out = self.model(src_signal, seg_film, mask.to(self.device))
                ref_signal = ref_signal.to(self.device)
                total_loss, cosin_loss, peak_to_peak_loss, mse_loss = self.criterion(out, ref_signal)
                val_loss_epoch += total_loss.item()
                self.val_loss_epoch = val_loss_epoch
                val_cosin_loss_epoch += cosin_loss.item()
                val_peak_to_peak_loss += peak_to_peak_loss.item()
                val_mse_loss += mse_loss.item()
            return val_loss_epoch / len(self.val_loader), val_cosin_loss_epoch / len(self.val_loader), val_peak_to_peak_loss / len(self.val_loader), val_mse_loss / len(self.val_loader)
    
    def test_epoch(self):
        self.model.eval()
        test_loss_epoch = 0
        test_cosin_loss_epoch = 0 
        test_peak_to_peak_loss = 0
        test_mse_loss = 0
        with torch.no_grad():
            for src_signal, ref_signal, seg_film, mask in tqdm(self.test_loader):
                src_signal = src_signal.to(self.device)
                seg_film = seg_film.to(self.device)
                out = self.model(src_signal, seg_film, mask.to(self.device))
                ref_signal = ref_signal.to(self.device)
                total_loss, cosin_loss, peak_to_peak_loss, mse_loss = self.criterion(out, ref_signal)
                test_loss_epoch += total_loss.item()
                test_cosin_loss_epoch += cosin_loss.item()
                test_peak_to_peak_loss += peak_to_peak_loss.item()
                test_mse_loss += mse_loss.item()
            return test_loss_epoch/ len(self.test_loader), test_cosin_loss_epoch / len(self.test_loader), test_peak_to_peak_loss/len(self.test_loader), test_mse_loss /  len(self.test_loader)

    def show_result(self, epoch):
        random_idx = random.choice(range(60))
        src_signal, ref_signal, seg_film, mask = self.test_loader.dataset[random_idx][0], self.test_loader.dataset[random_idx][1], self.test_loader.dataset[random_idx][2], self.test_loader.dataset[random_idx][3]
        src_signal = src_signal.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)
        out = self.model(src_signal.to(self.device), seg_film.unsqueeze(dim=0).to(self.device), mask.to(self.device)) 
        real_sr_signal = src_signal.clone()
        mask = mask.reshape(-1).cpu().detach().numpy()
        ref_signal = ref_signal.reshape(-1).cpu().detach().numpy()

        start, stop = depadding(ref_signal)
        
        plot_result(real_sr_signal.reshape(-1)[start:stop], out.reshape(-1).cpu().detach().numpy()[start:stop], ref_signal.reshape(-1)[start:stop])     
        self.tracking.log_image(image_data="src/experiments/results/result.jpg", name=f"Result at epoch: {epoch}")

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), "checkpoints/baseline.pth")
        if self.val_loss_epoch < self.BEST_LOSS:
            self.BEST_LOSS = self.val_loss_epoch
            torch.save(self.model.state_dict(), self.ckpt)
            self.tracking.log_model("model", self.ckpt)

    def training_experiment(self):
        print("BEGIN TRAINING ...")
        for epoch in range(1, self.epoch_n+1):
            with self.tracking.train():
                train_loss_epoch, train_cosin_loss_epoch, peak_to_peak_loss, train_mse_loss = self.train_epoch()
                self.tracking.log_metrics({
                    "total loss": train_loss_epoch,
                    "mse loss": train_mse_loss,
                    "cosin similarity": train_cosin_loss_epoch,
                    "peak-to-peak loss": peak_to_peak_loss
                }, epoch=epoch)

            with self.tracking.validate():
                val_loss_epoch, val_cosin_loss_epoch, peak_to_peak_loss, val_mse_loss = self.val_epoch()
                self.scheduler.step(val_loss_epoch)
                self.save_checkpoint()
                self.tracking.log_metrics({
                    "total loss": val_loss_epoch,
                    "mse loss": val_mse_loss,
                    "cosin similarity": val_cosin_loss_epoch,
                    "peak-to-peak loss": peak_to_peak_loss
                }, epoch=epoch)
            
            with self.tracking.test():
                test_loss_epoch, test_cosin_loss_epoch, test_peak_to_peak_loss, test_mse_loss = self.test_epoch()
                self.tracking.log_metrics({
                    "total loss": test_loss_epoch,
                    "mse loss": test_mse_loss,
                    "cosin similarity": test_cosin_loss_epoch,
                    "peak-to-peak loss": test_peak_to_peak_loss
                }, epoch=epoch)

            self.show_result(epoch)

            print("EPOCH: ", epoch, " - TRAIN_LOSS: ", train_loss_epoch, " || VAL_LOSS: ", val_loss_epoch, " || TEST_LOSS: ", test_loss_epoch)

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
import sys
sys.path.append(".")
from metrics.losses import CustomLoss, DisLoss, GenLoss
from src.dataloader.dataset import get_loader
from src.models.cpppg import Generator, Discriminator
from src.utils.utils import plot_result, depadding
import random
from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
from configs.seed import *


class AdversarialTrainer:
    def __init__(self, tracking, cfgs):
        self.tracking = tracking
        self.cfgs = cfgs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_value = self.cfgs['train']['clip_value']
        self.epoch_n = self.cfgs['train']['epoch_n']
        self.BEST_LOSS = np.inf
        self.ckpt = self.cfgs['train']['ckpt']
        self.enc = None 
        self.model = Generator(cfgs).to(self.device)   
        self.criterion = CustomLoss(self.cfgs).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), betas=(0.9, 0.999), weight_decay=0.005, lr=self.cfgs['train']['lr'])
        self.discriminator = Discriminator(cfgs).to(self.device) 
            
        if self.cfgs['train']['hinge']:
            self.gen_loss = GenLoss().to(self.device)
            self.discriminator_loss = DisLoss().to(self.device)
        else:
            self.discriminator_loss = torch.nn.BCELoss().to(self.device) 

        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), betas=(0.9, 0.999), weight_decay=0.005, lr=self.cfgs['train']['lr']/3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=8, verbose=True)
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.discriminator_optimizer, factor=0.9, patience=8, verbose=True)

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
        self.discriminator.train()
        train_loss_epoch = 0
        train_cosin_loss_epoch = 0
        train_peak_to_peak_loss = 0
        train_mse_loss = 0
        theta = 0.01

        for src_signal, ref_signal, seg_film, mask in tqdm(self.train_loader):
            
            src_signal = src_signal.to(self.device)
            seg_film = seg_film.to(self.device)
            out = self.model(src_signal, seg_film, mask.to(self.device))   

            self.optimizer.zero_grad()
            ref_signal = ref_signal.to(self.device)
            total_loss, cosin_loss, peak_to_peak_loss, mse_loss = self.criterion(out, ref_signal, self.enc)
            
            if self.cfgs['train']['hinge']:
                g_loss = self.gen_loss(self.discriminator(out))
            else:
                g_loss = self.discriminator_loss(self.discriminator(out), Variable(Tensor(src_signal.shape[0], 1).fill_(1.0), requires_grad=False))
            
            total_loss += theta * g_loss 
            train_loss_epoch += total_loss.item()
            train_cosin_loss_epoch += cosin_loss.item()
            train_peak_to_peak_loss += peak_to_peak_loss.item()
            train_mse_loss += mse_loss.item()
            total_loss.backward()
            self.optimizer.step()
            
            valid = Variable(Tensor(src_signal.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(src_signal.shape[0], 1).fill_(0.0), requires_grad=False)

            self.discriminator_optimizer.zero_grad()
            if self.cfgs['train']['hinge']:
                d_loss = self.discriminator_loss(self.discriminator(out.detach()), self.discriminator(ref_signal))

            else:
                real_loss = self.discriminator_loss(self.discriminator(ref_signal), valid)
                fake_loss = self.discriminator_loss(self.discriminator(out.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

            d_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()   

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
                total_loss, cosin_loss, peak_to_peak_loss, mse_loss = self.criterion(out, ref_signal, self.enc)
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
                total_loss, cosin_loss, peak_to_peak_loss, mse_loss = self.criterion(out, ref_signal, self.enc)
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
        plot_result(real_sr_signal.reshape(-1), out.reshape(-1).cpu().detach().numpy(), ref_signal.reshape(-1))     
        self.tracking.log_image(image_data="src/experiments/results/result.jpg", name=f"Result at epoch: {epoch}")
        
    def save_checkpoint(self):
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
                self.d_scheduler.step(val_loss_epoch)
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

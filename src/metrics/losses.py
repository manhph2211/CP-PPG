import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine
import sys 
sys.path.append(".")
from scipy.stats import skew
from src.utils.preprocess import get_peaks_info_of_segments, cycle_helper
import torch.nn.functional as F


class CustomLoss(nn.Module):
    
    def __init__(self, cfgs):
        super(CustomLoss, self).__init__()
        self.cfgs = cfgs
        self.beta = cfgs['metric']['beta']
        self.enc = cfgs['metric']['enc']
        self.alpha = cfgs['metric']['alpha']
        self.segment_length = cfgs['data']['segment_length']
    
    def cosine_similarity(self, rec_signal, ref_signal):
        dot_product = torch.sum(rec_signal * ref_signal, dim=-1)        
        original = torch.norm(rec_signal, dim=-1)
        reconstructed = torch.norm(ref_signal, dim=-1)        
        cosine_similarity = dot_product / (original * reconstructed)
        return torch.mean(cosine_similarity)

    def peak_to_peak_error(self, rec_signal, ref_signal):
        rec_peak_to_peak, ref_peak_to_peak = torch.FloatTensor(get_peaks_info_of_segments(rec_signal, ref_signal, return_ids=False, distance=10))
        return nn.MSELoss()(rec_peak_to_peak, ref_peak_to_peak)

    def peak_index_error(self, rec_signal, ref_signal):
        in_index, ref_index = torch.FloatTensor(get_peaks_info_of_segments(rec_signal, ref_signal, return_ids=True, distance=10))
        return nn.MSELoss()(in_index, ref_index)
    
    def custom_error(self, reconstructed, reference):
        batch_size = reconstructed.size(0)
        custom_loss = 0
        p2p_loss = 0

        for i in range(batch_size):
            in_signal = reconstructed[i].cpu().detach().numpy().reshape(-1)
            ref_signal = reference[i].cpu().detach().numpy().reshape(-1)
            in_peaks, _ = find_peaks(in_signal, distance=20)    
            ref_peaks, _ = find_peaks(ref_signal, distance=20)  
            # self.alpha = self.alpha # * (abs(len(in_peaks) - len(ref_peaks)))/len(ref_peaks))
            mse = nn.MSELoss()(reconstructed[i], reference[i])
            l1 = self.peak_to_peak_error(in_signal, ref_signal) 
            # l2 = self.peak_index_error(in_signal, ref_signal)
            if len(in_peaks) != len(ref_peaks):
                custom_loss += mse * self.alpha
            else:
                custom_loss += (l1.to(reconstructed.device)) * self.beta
                
            p2p_loss +=  (l1.to(reconstructed.device)) 
        
        return custom_loss, p2p_loss
    
    def encode_loss(self, reconstructed, reference, encoder=None):
        return nn.MSELoss()(encoder(reconstructed), encoder(reference)) if encoder is not None else 0

    def forward(self, in_signal, ref_signal, model=None):
        mse_loss = nn.MSELoss()(in_signal, ref_signal)
        cosine_similarity = self.cosine_similarity(in_signal, ref_signal)
        custom_loss, p2p_loss = self.custom_error(in_signal, ref_signal)
        total_loss =  mse_loss + custom_loss + self.enc * self.encode_loss(in_signal, ref_signal, model)
        return total_loss, cosine_similarity, p2p_loss, mse_loss


class DisLoss(nn.Module):
    def __init__(self):
        super(DisLoss, self).__init__()

    def forward(self, fake, real):
        return (F.relu(1 + fake) + F.relu(1 - real)).mean()
    

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()

    def forward(self, fake):
        return -fake.mean()


class AdaptiveMSELoss(nn.Module):
    def __init__(self):
        super(AdaptiveMSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        amplitude_ratio = torch.mean(torch.abs(y_true) / torch.abs(y_pred), dim=-1)
        base_loss = nn.MSELoss()(y_true, y_pred)
        adjusted_loss = amplitude_ratio * base_loss

        return adjusted_loss
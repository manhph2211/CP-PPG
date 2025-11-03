import numpy as np
import sys 
sys.path.append(".")
from src.utils.classification import classify
from src.utils.preprocess import read_processed_signal
from src.utils.utils import get_config, plot_two_signal
import glob
import random
import os
import numpy as np
import torch 
from scipy.interpolate import CubicSpline
from torchvision import transforms
from scipy.signal import filtfilt, resample_poly
from math import gcd
    

class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma=0.05):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        
    def forward(self, X):
        noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape)
        return X + noise


class Negation(torch.nn.Module):
    def __init__(self):
        super(Negation, self).__init__()
    def forward(self, X):
        return X * -1


class TimeFlip(torch.nn.Module):
    def __init__(self):
        super(TimeFlip, self).__init__()

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = X[:, ::-1].copy()
        elif torch.is_tensor(X):
            X = torch.flip(X, dims=[1])
        return X


class Scaling(torch.nn.Module):
    def __init__(self, sigma=0.1):
        super(Scaling, self).__init__()
        self.sigma = sigma
        
    def forward(self, X):
        scaling_factor = np.random.normal(loc=1.0, scale=self.sigma, size=(X.shape[0], 1))
        return X * scaling_factor


class TimeWarp(torch.nn.Module):
    def __init__(self, sigma=0.2, num_knots=4, num_splines=150):
        super(TimeWarp, self).__init__()
        self.sigma = sigma 
        self.num_knots = num_knots
        self.num_splines = num_splines

    def forward(self, X):
        X = np.expand_dims(X, axis=-1)
        time_stamps = np.arange(X.shape[1])
        knot_xs = np.arange(0, self.num_knots + 2, dtype=float) * (X.shape[1] - 1) / (self.num_knots + 1)
        spline_ys = np.random.normal(loc=1.0, scale=self.sigma, size=(self.num_splines, self.num_knots + 2))
    
        spline_values = np.array([self.get_cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])
    
        cumulative_sum = np.cumsum(spline_values, axis=1)
        distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)
    
        random_indices = np.random.randint(self.num_splines, size=(X.shape[0] * X.shape[2]))
    
        X_transformed = np.empty(shape=X.shape)
        for i, random_index in enumerate(random_indices):
            X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(time_stamps, distorted_time_stamps_all[random_index], X[i // X.shape[2], :, i % X.shape[2]])
        return X_transformed.squeeze(axis=-1)

    def get_cubic_spline_interpolation(self, x_eval, x_data, y_data):
        """
        Get values for the cubic spline interpolaßtion
        """
        cubic_spline = CubicSpline(x_data, y_data)
        return cubic_spline(x_eval)


class TSRandomCropPad(torch.nn.Module):
    """
        Crops a section of the sequence of a random length
        From the tsai package
    """
    def __init__(self, magnitude=0.05, ex=None, **kwargs):
        self.magnitude, self.ex = magnitude, ex
        super(TSRandomCropPad, self).__init__(**kwargs)
        
    def forward(self, o):
        o = torch.Tensor(o)
        if not self.magnitude or self.magnitude <= 0: return o
        seq_len = o.shape[-1]
        lambd = np.random.beta(self.magnitude, self.magnitude)
        lambd = max(lambd, 1 - lambd)
        win_len = int(round(seq_len * lambd))
        if win_len == seq_len: return o
        start = np.random.randint(0, seq_len - win_len)
        output = torch.zeros_like(o, dtype=o.dtype, device=o.device)
        output[..., start : start + win_len] = o[..., start : start + win_len]
        if self.ex is not None: output[...,self.ex,:] = o[...,self.ex,:]
        return output
        
        
class ResampleSignal(torch.nn.Module):
    def __init__(self, fs_original, fs_target):
        super(ResampleSignal, self).__init__()
        self.fs_original = fs_original
        self.fs_target = fs_target
        
    def forward(self, X):
        X = X.squeeze()
        up, down = self.get_sampling_factor()
        X = resample_poly(X, up, down)
        return np.expand_dims(X, axis=0)

    def get_sampling_factor(self):
        gcd_value = gcd(self.fs_original, self.fs_target)
        up = self.fs_target // gcd_value
        down = self.fs_original // gcd_value
        return up, down


class PPGTransform:
    def __init__(self, probability: float = 0.8):
        self.probability = probability

    def time_warp(self, signal: np.ndarray, factor: float = random.uniform(0.01, 0.02)) -> np.ndarray:
        if np.random.rand() > self.probability:
            return signal
        
        time_steps = np.arange(len(signal))
        stretched_time = time_steps + np.random.uniform(-factor, factor) * time_steps
        return np.interp(time_steps, stretched_time, signal)

    def amplitude_scaling(self, signal: np.ndarray, factor: float = random.uniform(0.01, 0.02)) -> np.ndarray:
        if np.random.rand() > self.probability:
            return signal
        
        scaling_factor = 1 + np.random.uniform(-factor, factor)
        return signal * scaling_factor

    def shifting(self, signal, shift=None):
        if np.random.rand() > self.probability:
            return signal
        
        shift = int(np.round(np.random.uniform(0, 3)))
            
        shifted_signal = np.roll(signal, shift)
        if shift < 0:
            shifted_signal[shift:] = 0 
        elif shift > 0:
            shifted_signal[:shift] = 0 
        return shifted_signal

    def jittering(self, signal: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        if np.random.rand() > self.probability:
            return signal
        
        noise = np.random.normal(0, np.abs(noise_factor * np.mean(signal)), len(signal))
        return signal + noise

    def baseline_drift(self, signal: np.ndarray, factor: float = random.uniform(0.01, 0.02)) -> np.ndarray:
        if np.random.rand() > self.probability:
            return signal
        
        drift = factor * np.interp(np.linspace(0, 10, len(signal)), 
                                   np.linspace(0, 10, 10), 
                                   np.random.rand(10) - 0.5)
        return signal + drift

    def convert(self, signal: np.ndarray) -> np.ndarray:
        signal = self.time_warp(signal)
        signal = self.shifting(signal)
        signal = self.amplitude_scaling(signal)
        signal = self.jittering(signal)
        signal = self.baseline_drift(signal)
        return signal

    def transform(self, segments):
        new_segments = []
        for segment in segments:
            new_segments.append(self.convert(segment).reshape(1,-1))
        return np.array(new_segments)
    
    
if __name__=="__main__":
    
    def prepare_in_data(cfgs):
        segments = []
        pressures = []
        for csv_file in glob.glob(os.path.join(cfgs['data']['root']['train_val'], "*.csv")):
            seg_in, _, _ = read_processed_signal(csv_file, resample=cfgs['data']['segment_length'], smooth=cfgs['data']['smooth'])
            segments.extend(seg_in)
        return np.array(segments).reshape(-1,seg_in.shape[-1]), np.array(pressures).reshape(-1,seg_in.shape[-1])

    in_seg, _ = prepare_in_data(get_config())
    transform = PPGTransform(probability=0.5)
    for i in range(3):
        signal_segment = in_seg[random.choice(range(len(in_seg)))]
        transformed_signal = transform.convert(signal_segment)
        plot_two_signal(signal_segment, transformed_signal)

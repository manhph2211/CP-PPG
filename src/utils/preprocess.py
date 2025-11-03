import pandas as pd
import scipy
from scipy import interpolate, signal
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import glob
import os
import sys
import math
from collections import defaultdict
sys.path.append(".")
from src.utils.classification import classify
from src.utils.utils import plot_wavform, get_feet, get_peaks_info, depadding
from configs.seed import *
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm
import pandas as pd
from pyampd.ampd import find_peaks

import scipy.signal
from scipy.signal import correlate
from scipy.interpolate import CubicSpline


def peakfind(x, distance=40, prominence=200):
    peaks, _ = scipy.signal.find_peaks(x, distance=distance, prominence=prominence)
    return peaks


def get_scaler(segments):
    src_segments, ref_segments = segments[:,0,:,:], segments[:,1,:,:]
    src_scaler = MinMaxScaler()
    ref_scaler = MinMaxScaler()

    src_scaler.fit(src_segments.reshape(-1,1))
    joblib.dump(src_scaler, "assets/src_scaler.pkl")

    ref_scaler.fit(ref_segments.reshape(-1,1))
    joblib.dump(ref_scaler, "assets/ref_scaler.pkl") 
    scaler = [src_scaler, ref_scaler]
    return scaler


def process_raw_signal(csv_file): 
    '''raw data file path from raw folder''' 
    df = pd.read_csv(csv_file)
    peaks = peakfind(df['PPG_In_LPF_10Hz'].values, distance=40, prominence=10)
    line = interpolate.CubicSpline(df['time'].values[peaks], df['PPG_In_LPF_10Hz'].values[peaks])
    df['PPG_In_AC'] = (df['PPG_In_LPF_10Hz'] - line(df['time'].values))*-1
    peaks = peakfind(df['PPG_Ref_LPF_10Hz'].values)
    line = interpolate.CubicSpline(df['time'].values[peaks], df['PPG_Ref_LPF_10Hz'].values[peaks])
    df['PPG_Ref_AC'] = (df['PPG_Ref_LPF_10Hz'] - line(df['time'].values))*-1
    folder_name = csv_file.split("/")[-2]
    if not os.path.exists("/".join(csv_file.replace(folder_name,"processed").split("/")[:-1])):
        os.makedirs(csv_file.replace(folder_name,"processed")[:-4])
    df.to_csv(csv_file.replace(folder_name,"processed"))


def moving_average(signal, window_size=5):
    if window_size % 2 == 0: 
        window_size += 1 
    half_window = window_size // 2
    smoothed = []
    for i in range(len(signal)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(signal), i + half_window + 1)
        window = signal[start_idx:end_idx]
        smoothed.append(sum(window) / len(window))
    return smoothed


def read_processed_signal(csv_file, cfgs):
    data_version = cfgs['data']['root']['train_val'].split("/")[-1]
    sampling_rate = cfgs['data']['sampling_rate']
    raw_data = genfromtxt(csv_file, delimiter=',') 
    if data_version == "v2": 
        raw_data = raw_data[1:,-2:] # v2
    elif data_version == "v1":
        raw_data = raw_data[1:,-3:-1] # v1
    else:
        raise ValueError("Wrong data version")

    raw_data = raw_data.reshape((-1,2))
    ppg_in, ppg_ref, pressure = raw_data[:,0], raw_data[:,1], raw_data[:,0]  #sensor_split(raw_data)
    if cfgs['data']['window'] > 1:
        new_length = int(len(ppg_in) * 128 /sampling_rate)
        ppg_in, ppg_ref, pressure = signal.resample(ppg_in, new_length), signal.resample(ppg_ref, new_length), signal.resample(pressure, new_length)
    return ppg_in, ppg_ref, pressure


def read_indices(csv_file, cfgs):
    raw_data = pd.read_csv(csv_file)
    start_ids = raw_data['ref_s'].values
    end_ids = raw_data['ref_e'].values
    ref_types = raw_data['ref_t'].values
    src_types = raw_data['in_t'].values
    return start_ids, end_ids, ref_types, src_types


def custom_segments(data, window, stride):
    num_samples, _, segment_len = data.shape

    num_segments = ((num_samples - window) // stride) + 1

    segments = []

    for i in range(num_segments):
        start = i * stride
        end = start + window
        segments.append(data[start:end,:,:].reshape(-1, window*segment_len))

    return np.array(segments)


def extract_window_segments(csv_file, cfgs, window_size=None):
    ppg_in, ppg_ref, _ = read_processed_signal(csv_file, cfgs)
    sampling_rate = cfgs['data']['sampling_rate']
    if window_size is None:
        window_size = cfgs['data']['window'] * sampling_rate
    stride = cfgs['data']['stride'] * sampling_rate
    overlap = window_size - stride
    src_segments = []
    ref_segments = []

    for i in tqdm(range(0, len(ppg_in) - window_size + 1, overlap)):
        src_segments.append(ppg_in[i:i+window_size])
        ref_segments.append(ppg_ref[i:i+window_size])

    src_segments = np.array(src_segments)
    ref_segments = np.array(ref_segments)
    
    return src_segments, ref_segments


def cycle_helper(src_signal, ref_signal, return_cycles=False):
    feet = get_feet(ref_signal)
    if len(feet) <= 2:
        return None, None, None
    
    updated_window_in = [] 
    breakdown_seg_in = [] 
    updated_window_ref = []
    breakdown_seg_ref = []

    updated_window_in.extend(src_signal[:feet[0]])
    breakdown_seg_in.append(src_signal[:feet[0]].tolist())
    updated_window_ref.extend(ref_signal[:feet[0]])
    breakdown_seg_ref.append(ref_signal[:feet[0]].tolist())

    for j in range(len(feet)-1):
        start_idx = feet[j]
        end_idx = feet[j + 1]
        cycle_in = src_signal[start_idx:end_idx]
        cycle_ref = ref_signal[start_idx:end_idx]
        updated_window_in.extend(cycle_in)
        breakdown_seg_in.append(cycle_in.tolist())
        updated_window_ref.extend(cycle_ref)
        breakdown_seg_ref.append(cycle_ref.tolist())

    updated_window_in.extend(src_signal[feet[-1]:])
    breakdown_seg_in.append(src_signal[feet[-1]:].tolist())
    updated_window_ref.extend(ref_signal[feet[-1]:])
    breakdown_seg_ref.append(ref_signal[feet[-1]:].tolist())


    updated_window_in = np.array(updated_window_in)
    updated_window_ref = np.array(updated_window_ref)

    if return_cycles:
        return breakdown_seg_in, breakdown_seg_ref, None
    return updated_window_in, updated_window_ref, None


def get_peaks_info_of_segments(src_signal_, ref_signal_, return_ids=True, return_ratio=False, distance=None):
    try:
        src_signal = src_signal_.copy().detach().numpy().reshape(-1)
        ref_signal = ref_signal_.copy().detach().numpy().reshape(-1)
    except:
        src_signal = src_signal_.copy().reshape(-1)
        ref_signal = ref_signal_.copy().reshape(-1)

    ids_in, ids_ref, amplitudes_in, amplitudes_ref = [], [], [], []

    breakdown_seg_in, breakdown_seg_ref, _ = cycle_helper(src_signal, ref_signal, return_cycles=True)
        
    for cycle_in, cycle_ref in zip(breakdown_seg_in, breakdown_seg_ref):
  
        amplitude_ref, id_ref = get_peaks_info(cycle_ref, return_ratio, distance=distance)
        amplitude_in, id_in = get_peaks_info(cycle_in, return_ratio, distance=distance)

        if id_ref is None or id_in is None:
            continue 

        amplitudes_ref.append(amplitude_ref)
        ids_ref.append(id_ref)     

        amplitudes_in.append(amplitude_in)
        ids_in.append(id_in)

    if return_ids:
        return np.array(ids_in), np.array(ids_ref)
    else:
        return np.array(amplitudes_in), np.array(amplitudes_ref)
    

def normalize_data(x):
    return (x - x.min()) / (x.max() - x.min())


def waveform_norm(x):
    return (x - x.min())/(x.max() - x.min() + 1e-6)


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = fs * 0.5  
    lowcut = lowcut / nyq  
    highcut = highcut / nyq
    b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', analog=False)
    return scipy.signal.filtfilt(b, a, data)


def remove_mean(data):
    return data-np.mean(data)


def mean_filter_normalize(data, fs=128, lowcut=0.5, highcut=8, order=1, use_norm=True):
    data = data-np.mean(data)
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    if use_norm:
        data = normalize_data(data)
    
    return data


def align_pair(abp, raw_ppg, windowing_time, fs):
    window_size = fs * windowing_time 
    extract_size = fs * (windowing_time-1)

    cross_correlation = correlate(abp, raw_ppg)
    shift = np.argmax(cross_correlation[extract_size:window_size])
    shift += extract_size
    start = np.abs(shift-window_size)

    a_abp = abp[:extract_size]
    a_rppg = raw_ppg[start:start+extract_size]

    return a_abp, a_rppg, shift-window_size


def rm_baseline_wander(ppg, vlys, add_pts = True):
    rollingmin_idx = vlys
    rollingmin = ppg[vlys]
    
    mean = np.mean(rollingmin)
    
    if add_pts == True:
        dist = np.median(np.diff(rollingmin_idx))
        med = np.median(rollingmin)
        
        add_pts_head = math.ceil(rollingmin_idx[0] / dist)
        head_d = [rollingmin_idx[0]-i*dist for i in reversed(range(1,add_pts_head+1))] 
        head_m = [med]*add_pts_head
        
        
        add_pts_tail = math.ceil((len(ppg)-rollingmin_idx[-1]) / dist)
        tail_d = [rollingmin_idx[-1]+ i*dist for i in range(1,add_pts_tail+1)] 
        tail_m = [med]*add_pts_tail 
        
        rollingmin_idx = np.concatenate((head_d, rollingmin_idx, tail_d))
        rollingmin = np.concatenate((head_m, rollingmin, tail_m))

    cs = CubicSpline(rollingmin_idx, rollingmin)
    baseline = cs(np.arange(len(ppg)))    
    rem_line = ppg - (baseline-mean)
    return rem_line, baseline, rollingmin, rollingmin_idx


def identify_out_pk_vly(sig, pk, vly, th=3):
    outs = []
    
    vly_val = sig[vly]
    pk_val = sig[pk]
    
    vly_val_idx = vly_val.argmin()
    vly_val_min = vly_val[vly_val_idx]
    vly_val_argmin = vly[vly_val_idx]
    vly_val_mean = vly_val.mean()
    vly_val_std = vly_val.std()
    
    if vly_val_min < vly_val_mean - vly_val_std * th:
        outs.append(vly_val_argmin)
    
    pk_val_idx = pk_val.argmax()
    pk_val_max = pk_val[pk_val_idx]
    pk_val_argmax = pk[pk_val_idx]
    pk_val_mean = pk_val.mean()
    pk_val_std = pk_val.std()
    
    if pk_val_max > pk_val_mean + pk_val_std * th:
        outs.append(pk_val_argmax)
    
    return outs


import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import neurokit2 as nk
import sys
sys.path.append(".")
from configs.seed import *
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.style.use("opinionated_m")
import colormaps as cmaps
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.grid'] = False  


def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def write_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def standardize(x):
    return (x - x.min()) / (x.max() - x.min())


def get_config(yaml_file='configs/config.yml'):
    with open(yaml_file, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    return cfgs


def get_feet(signal, threshold=30):
    inverted_signal = -signal
    feet, _ = find_peaks(inverted_signal, distance=64)
    z_scores = np.abs((feet - np.mean(feet)) / np.std(feet))
    feet = feet[(z_scores < threshold)]
    return feet


def get_peaks_info(signal, return_ratio=False, distance=None):
    '''
    Apply only for 1 cycle only
    '''
    try:
        signal = signal.cpu().detach().numpy().reshape(-1)
    except:
        signal = np.array(signal).reshape(-1)
    assert signal.shape[0] > 0
    # plot_two_signal(signal, signal)

    peaks, _ = find_peaks(signal, distance=distance)
    
    if len(peaks) < 1:
        amplitudes = None
        ids = None
    elif len(peaks) == 1:
        if not return_ratio:
            amplitudes = [signal[peaks[0]], signal[peaks[0]], signal[peaks[0]]]
        else:
            amplitudes = [1, 1, 1]
        ids = [peaks[0]/len(signal), peaks[0]/len(signal), peaks[0]/len(signal)]
    else:
        dicrotic_notch_idx = np.argmin(signal[peaks[0]:peaks[1]]) + peaks[0]
        if not return_ratio:
            amplitudes = [signal[peaks[0]], signal[dicrotic_notch_idx], signal[peaks[1]]]
        else:
            amplitudes = [signal[peaks[0]]/np.max(signal), signal[dicrotic_notch_idx]/np.max(signal), signal[peaks[1]]/np.max(signal)]
        ids = [peaks[0]/len(signal), dicrotic_notch_idx/len(signal), peaks[1]/len(signal)]
        
    return amplitudes, ids


def depadding(mask):
    """ Helper Function: Mask should be a numpy array or torch tensor, with padding at the end
    """
    mask = mask.reshape(-1)
    if isinstance(mask, torch.Tensor):
        mask = (mask != mask[-1]).to(torch.float32).to(mask.device)
        non_zero_idx = torch.where(mask)[0]
    else:
        mask = (mask != mask[-1]).astype(float)
        non_zero_idx = np.where(mask)[0]  
    start = non_zero_idx[0].item() if len(non_zero_idx) > 0 else 0
    stop = non_zero_idx[-1].item() + 1 if len(non_zero_idx) > 0 else mask.shape[0]

    return start, stop


def plot_subject_distribution(subjects_dict, label="Number of Segments"):
    subjects = list(subjects_dict.keys())
    segment_counts = list(subjects_dict.values())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(subjects, segment_counts, color='skyblue')
    ax.set_xlabel('Subject Name', labelpad=15, fontsize=18) 
    ax.set_ylabel(label, labelpad=15, fontsize=18) 
    ax.set_title('Subject Distribution', pad=20, fontsize=20) 
    ax.set_xticks(range(len(subjects))) 
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    plt.show()

    fig.savefig('subject_distribution.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight')
    
    
def plot_wavform(src_ppg_signal, ref_ppg_signal, label=None, name=None):
    assert len(src_ppg_signal) == len(ref_ppg_signal)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(src_ppg_signal)
    ax1.set_title("Source Signal")

    ax2.plot(ref_ppg_signal)
    ax2.set_title("Reference Signal")

    plt.tight_layout()
    
    if label is not None:
        fig.suptitle(label, y=1.05)

    if name is not None:
        plt.savefig(f"trash/images/{name}.jpg")
    else:
        plt.savefig(f"src/experiments/results/result.jpg")        
    plt.show()


def plot_result(src_ppg_signal, out_ppg_signal, ref_ppg_signal, ppg_label=None):
    '''
    Note: This function is to visualize source, predicted and referece signal in one figure.
    '''
    out_ppg_signal = standardize(out_ppg_signal)
    if ref_ppg_signal is None: 
        save_file = "src/experiments/results/noise.jpg"
        ref_ppg_signal = out_ppg_signal
    else:
        save_file = "src/experiments/results/result.jpg"
        assert len(src_ppg_signal) == len(ref_ppg_signal) == len(out_ppg_signal)

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    axes.plot(src_ppg_signal, color='blue', label='Source Signal')
    axes.plot(out_ppg_signal, color='green', label='Reconstructed Signal')
    axes.plot(ref_ppg_signal, color='red', label='Reference Signal')
    axes.set_xlabel("Sample")
    axes.set_ylabel("Amplitude")
    axes.legend()
    plt.tight_layout()
    
    if ppg_label is not None:
        plt.show()
    plt.savefig(save_file)
    fig.savefig('Waveforms.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight')

    
def plot_peaks(signal):
    peaks, _ = find_peaks(signal)    
    feet = get_feet(signal)
    cycle_peaks = []
    cycle_notches = []
    cycle_feet = []
    for i in range(len(feet) - 1):
        start_idx = feet[i]
        end_idx = feet[i + 1]

        cycle_peaks_in_range = [peak for peak in peaks if start_idx < peak < end_idx]
        cycle_notches_in_range = [notch for notch in find_peaks(-signal[start_idx:end_idx])[0]]

        cycle_peaks.extend(cycle_peaks_in_range)
        cycle_notches.extend([notch + start_idx for notch in cycle_notches_in_range])
        cycle_feet.extend([start_idx, end_idx])

    plt.figure(figsize=(24, 5))
    plt.plot(signal, label="Signal", color='blue')
    plt.plot(cycle_peaks, signal[cycle_peaks], "x", label="Peaks", color='red')
    plt.plot(cycle_notches, signal[cycle_notches], "o", label="Notches", color='green')
    plt.plot(cycle_feet, signal[cycle_feet], "v", label="Feet", color='purple')
    plt.legend()
    plt.title("Signal with Peaks, Notches, and Feet")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


def plot_two_signal(signal_1, signal_2, label1="Input Signal", label2="Augmented Signal"):    
    plt.figure(figsize=(24,5))
    plt.plot(signal_1, label=f"{label1}", color='blue')
    plt.plot(signal_2, label=f"{label2}", color='red')
    plt.legend()
    plt.show()


def plot_metrics(signal1, signal2):
    def find_relevant_points(signal):
        peaks, _ = find_peaks(signal)
        notches, _ = find_peaks(-signal)
        
        peak1, peak2 = peaks[:2]
        relevant_notch = notches[(notches > peak1) & (notches < peak2)][0]

        plt.scatter([peak1, peak2], [signal[peak1], signal[peak2]], color='brown')
        plt.scatter(relevant_notch, signal[relevant_notch], color='pink', marker='x')

        plt.plot([peak1, peak1], [0, signal[peak1]], color='blue', linestyle='--')
        plt.plot([relevant_notch, relevant_notch], [0, signal[relevant_notch]], color='red', linestyle='--')
        plt.plot([peak2, peak2], [0, signal[peak2]], color='orange', linestyle='--')

        plt.plot([peak1, 0], [signal[peak1], signal[peak1]], color='blue', linestyle='--')
        plt.plot([relevant_notch, 0], [signal[relevant_notch], signal[relevant_notch]], color='red', linestyle='--')
        plt.plot([peak2, 0], [signal[peak2], signal[peak2]], color='orange', linestyle='--')

    plt.figure(figsize=(12, 6))
    plt.plot(signal1, label="Signal 1", color='green')
    plt.plot(signal2, label="Signal 2", color='purple')
    
    find_relevant_points(signal1)
    find_relevant_points(signal2)

    plt.legend()
    plt.title("Comparison of Two Signals based on Peaks and Notches")
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitudes')
    plt.show()

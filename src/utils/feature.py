import numpy as np
from pyampd.ampd import find_peaks
import scipy.signal
import pyampd
import matplotlib.pyplot as plt 
import numpy as np
from pyampd.ampd import find_peaks
import scipy.signal
import copy
from src.utils.utils import *
from src.utils.preprocess import *


def _is_flat(x, threshold=0.3):
    delta=1e-5
    dx = x[1:] - x[:-1]
    flat_parts = (np.abs(dx) < delta).astype("int").sum()
    flat_parts = flat_parts/x.shape[0]    
    return flat_parts > threshold


def skew(x, is_flat_threshold): 
    return np.sum(((x - x.mean())/(x.std()+1e-6))**3)/x.shape[0] * ~_is_flat(x, threshold=is_flat_threshold)


def kurtosis(x): 
    sqi_score = scipy.stats.kurtosis(x)

    if np.isnan(sqi_score) or np.isinf(sqi_score):
        return 0.0

    return sqi_score


def width_at_per(per, cycle, peak, fs):
    height_to_reach = cycle[peak]*per
    i = 0 
    while i < peak and cycle[i] < height_to_reach:
        i+=1
    SW = peak - i
    i = peak
    while i < len(cycle) and cycle[i] > height_to_reach:
        i +=1
    i -= 1
    DW = i - peak
    return SW, DW 
    

def vpg_points(vpg, peak):
    pks = find_peaks(vpg)
    vlys = find_peaks(-vpg)    
    Tsteepest = np.argmax(vpg[:peak])
    w = Tsteepest 
    
    pks = pks[pks > peak]
    vlys = vlys[vlys > peak]

    if len(vlys) < 1:
        end = int((len(vpg)-peak)*0.4)+peak
        y = np.argmin(vpg[peak:end])+peak
    else:
        y = vlys[0]
        
    min_slope_idx = y
    end = int((len(vpg)-min_slope_idx)*0.4)+min_slope_idx
    TdiaRise = np.argmax(vpg[min_slope_idx:end])+min_slope_idx
    z = TdiaRise
    return w, y, z
   

def apg_points(apg, peak, w, y, z):
    a = b = c = d = e = 0    
    pks = find_peaks(apg[:int(len(apg)*0.6)])
    vly, _ = scipy.signal.find_peaks(-apg[:int(len(apg)*0.6)])
    
    if len(pks) < 1 or len(vly) < 1: 
        return a, b, c, d, e
    
    a = pks[0]
    if a > peak:
        a = np.argmax(apg[0:peak])
    else:
        pks = pks[1:]
        
    vly = vly[vly > a]
    if len(vly) < 1: 
        return a, b, c, d, e
    
    vly = vly[vly > w]
    if len(vly) < 1: 
        return a, b, c, d, e

    pks_tmp = pks[pks > y] 
    if len(pks_tmp)!=0:
        e = pks_tmp[np.argmax(apg[pks_tmp])]
    else: 
        e =  np.argmax(apg[y+1:])
    pks = pks[pks < e]
    vly = vly[vly < e]
    
    if len(vly[vly < peak-2]) < 1:
        vly_b, _ = scipy.signal.find_peaks(-apg[w:peak-2])
        vly_b+=w
        
        if len(vly_b) < 1:
            b = np.argmin(apg[w:peak-2])+w
        else:
            b = vly_b[0]
    else:
        b = vly[0]
    
    vly_tmp, _ = scipy.signal.find_peaks(-apg[peak:e])
    vly_tmp+=peak
    if len(vly_tmp) < 1:
        d = np.argmin(apg[peak:e])+peak
    else:
        d = vly_tmp[np.argmin(apg[vly_tmp])]

    pks_tmp, _ = scipy.signal.find_peaks(apg[b:d+1])
    pks_tmp+=b
    if len(pks_tmp) < 1:
        c = np.argmax(apg[b:d])+b
    else:
        c = pks_tmp[np.argmax(apg[pks_tmp])]
    
    return a, b, c, d, e


def extract_apg_feat(cycle, vpg, peak, w, y, z, fs):   
    feats = []
    feats_header = []

    apg = np.diff(vpg)
    
    Tc = len(cycle)
    
    a, b, c, d, e = apg_points(apg,peak,w, y, z)
    apg_p = [a, b, c, d, e]
    apg_p_names = ['a', 'b', 'c', 'd', 'e']
    
    feats += [apg[i] for i in apg_p]
    feats_header += ['apg_'+i for i in apg_p_names]
    
    feats += [cycle[i+2] for i in apg_p]
    feats_header += ['ppg_'+i for i in apg_p_names]
    
    feats += [apg[i]/apg[a] for i in apg_p[1:]]
    feats_header += ['ratio_apg_'+i for i in apg_p_names[1:]]
    
    feats += [cycle[i+2]/cycle[a+2] for i in apg_p[1:]]
    feats_header += ['ratio_ppg_'+i for i in apg_p_names[1:]]
    
    feats += [a/fs,(b-a)/fs,(c-b)/fs,(d-c)/fs,(e-d)/fs]
    feats_header += ['T_'+i for i in apg_p_names]
    
    feats += [a/Tc,(b-a)/Tc,(c-b)/Tc,(d-c)/Tc,(e-d)/Tc]
    feats_header += ['T_'+i+'_norm' for i in apg_p_names]
    
    feats += [(peak-a)/fs,(peak-b)/fs,(c-peak)/fs,(d-peak)/fs,(e-peak)/fs]
    feats_header += ['T_peak_'+i for i in apg_p_names]
    
    feats += [(peak-a)/Tc,(peak-b)/Tc,(c-peak)/Tc,(d-peak)/Tc,(e-peak)/Tc]
    feats_header += ['T_peak_'+i+'_norm' for i in apg_p_names]
    
    feats += [(apg[b]-apg[c]-apg[d]-apg[e])/apg[a]]
    feats_header += ['AI']
    
    bd = (vpg[d+1] - vpg[b+1])/(d-b)
    bcda = (apg[b] - apg[c] - apg[d])/apg[a]
    sdoo = np.sum(vpg[peak:d+1]*vpg[peak:d+1])/np.sum(vpg*vpg)
    
    feats += [bd, bcda, sdoo]
    feats_header += ['bd', 'bcda', 'sdoo']
    
    return feats_header, feats


def extract_temp_feat(cycle, peak, fs):
    feat = []
    Tc = len(cycle)
    Ts = peak    
    Td = len(cycle) - peak
    min_val=np.min(cycle)
    AUCsys = np.trapz(cycle[:peak]-min_val)
    AUCdia = np.trapz(cycle[peak:]-min_val)
    
    point_feat_name = ['Tc', 'Ts', 'Td', "sys_amp"]
    point_feat = [Tc/fs, Ts/fs, Td/fs, cycle[peak]]

    width_names = []
    width_feats = []

    for per in [0.25,0.50,0.75]:
        SW, DW = width_at_per(per, cycle, peak, fs)
        per_str = str(int(per*100))
        width_names += ['SW'+per_str, 'SW'+per_str+'_norm', 
                               'DW'+per_str, 'DW'+per_str+'_norm', 
                               'SWaddDW'+per_str, 'SWaddDW'+per_str+'_norm']
        width_feats +=[SW/fs, SW/Tc,
                        DW/fs, DW/Tc,
                        (SW+DW)/fs, (SW+DW)/Tc]

    point_feat_name += width_names
    point_feat += width_feats
    
    area_feat_name = ['AUCsys','AUCdia']
    area_feat = [AUCsys,AUCdia]
        
    feat_name = point_feat_name + area_feat_name
    feat = point_feat + area_feat 
    
    SQI_skew = skew(np.array(cycle),0.3)
    SQI_kurtosis = kurtosis(np.array(cycle))
    feat_name = feat_name + ["skewness", "kurtosis"]
    feat = feat + [SQI_skew, SQI_kurtosis]    
    return np.array(feat_name), np.array(feat)


def extract_feat_cycle(cycles, fs):
    
    feats = []
    feat_name = []

    dia_feats = []
    dia_feat_name = []
    valid = 0

    dia_feat_name = ["notch_amp","notch_idx","dia_amp","dia_idx","p2p_time"]

    for c in cycles:
        peaks, _ = scipy.signal.find_peaks(c, distance=20)
        if len(peaks) == 0:
            continue

        feat_name,feat= extract_temp_feat(c, peaks[0], fs)
        feats.append(feat)

        if len(peaks) > 1:
            valid += 1
            dia_peak = peaks[1]
            dicrotic_notch_idx = np.argmin(c[peaks[0]:peaks[1]]) + peaks[0]
            dia_feats += [[c[dicrotic_notch_idx],dicrotic_notch_idx/len(c), c[dia_peak], dia_peak/len(c), (dia_peak-peaks[0])/len(c)]]

    if len(dia_feats)>0: dia_feats = np.vstack(dia_feats).mean(axis=0)
    else: dia_feats = np.array([])

    feats = np.vstack(feats).mean(axis=0)
        
    return feat_name, feats, dia_feat_name, dia_feats, valid


def _compute_cyle_pks_vlys(sig, fs=128, pk_th=0.6, remove_start_end = True):
    peaks = find_peaks(sig, scale=int(fs))
    valleys = find_peaks(sig.max()-sig, scale=int(fs))

    flag1, flag2 = False, False
    
    if peaks[0] == 0: peaks = peaks[1:]
    if valleys[0] == 0: valleys = valleys[1:]
    if peaks[-1] == len(sig)-1: peaks = peaks[:-1]
    if valleys[-1] == len(sig)-1: valleys = valleys[:-1]

    if len(peaks)==0 or len(valleys)==0:
        return True, True, [], []
        
    if remove_start_end:
        if peaks[0] < valleys[0]: peaks = peaks[1:]
        else: valleys = valleys[1:]

        if peaks[-1] > valleys[-1]: peaks = peaks[:-1]
        else: valleys = valleys[:-1]

    if len(peaks)==0 or len(valleys)==0:
        return True, True, [], []
        
    while len(peaks)!=0 and peaks[0] < valleys[0]:
        peaks = peaks[1:]
    
    while len(peaks)!=0 and peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
    
    if len(peaks)==0 or len(valleys)==0:
        return True, True, [], []
        
    new_peaks = []
    mean_vly_amp = np.mean(sig[valleys])
    
    for i in range(len(peaks)-1):
        if sig[peaks[i]]-mean_vly_amp > (sig[peaks[i+1]]-mean_vly_amp)*pk_th:
            new_peaks.append(peaks[i])
            a=i
            break
            
    if len(peaks) == 1:
        new_peaks.append(peaks[0])
        a=0

    for j in range(a+1,len(peaks)):
        if sig[peaks[j]]-mean_vly_amp > (sig[new_peaks[-1]]-mean_vly_amp)*pk_th:
            new_peaks.append(peaks[j])
            
    if not np.array_equal(peaks,new_peaks):
        flag1 = True
        
    if len(valleys)-1 != len(new_peaks):
        flag2 = True
        
    if len(valleys)-1 == len(new_peaks):
        for i in range(len(valleys)-1):
            if not(valleys[i] < new_peaks[i] and new_peaks[i] < valleys[i+1]):
                flag2 = True
        
    return flag1, flag2, new_peaks, valleys


def extract_cycle_check(sig, fs, pk_th=0.6, remove_start_end = True):

    flag1, flag2, new_peaks, valleys = _compute_cyle_pks_vlys(sig, fs, pk_th=pk_th, remove_start_end = remove_start_end)

    cycles = []
    peaks_norm = []

    if len(new_peaks)!=0 and len(valleys) !=0:
        ## Save segments
        for i in range(len(valleys)-1):
            #print((valleys[i],valleys[i+1]))
            cycles.append(sig[valleys[i]:valleys[i+1]])
            
        ## Save peaks
        if len(valleys)-1 == len(new_peaks):
            for i in range(len(new_peaks)):
                peaks_norm.append(new_peaks[i]-valleys[i])
    
    return cycles, peaks_norm, flag1, flag2, new_peaks, valleys


def extract_feat_cycle_peaks_norm(cycles, peaks_norm, fs):
    
    feats = []
    feat_name = []

    for c, p in zip(cycles, peaks_norm):
        try:
            feat_name,feat= extract_temp_feat(c, p, fs)
            feats.append(feat)
        except:
            print("Cycle ignored")

    if len(feats)>0: feats = np.vstack(feats).mean(axis=0)
    else: feats = np.array([])
        
    return feat_name, feats


def extract_feat_original(sig, fs, filtered=True, remove_start_end=True):
    ppg = PPG(sig,fs)
    _, head, feat_str = ppg.features_extractor(filtered=filtered, remove_first=remove_start_end)
    
    feat = [float(s) for s in feat_str.split(', ')]
    
    return head, feat


def extract_temp_feat_all(cycle, peak, fs):
    feat = []    
    Tc = len(cycle)
    Ts = peak
    
    Td = len(cycle) - peak
    
    vpg = np.diff(cycle)
    w, y, z = vpg_points(vpg, peak)
    
    Tsteepest = w
    Steepest = vpg[w]
    
    TNegSteepest = y
    
    TdiaRise = z
        
    NegSteepest = vpg[TNegSteepest]
    
    DiaRise = cycle[TdiaRise]
    SteepDiaRise = vpg[TdiaRise]
    
    TSystoDiaRise = TdiaRise - Ts
    
    TdiaToEnd = Tc - TdiaRise
    
    Ratio = cycle[peak]/DiaRise
    
    point_feat_name = ['Tc', 'Ts', 'Td', 'Tsteepest', 'Steepest', 'TNegSteepest', 'NegSteepest', 
            'TdiaRise', 'DiaRise', 'SteepDiaRise', 'TSystoDiaRise', 'TdiaToEnd', 'Ratio']
    point_feat = [Tc/fs, Ts/fs, Td/fs, Tsteepest/fs, Steepest, TNegSteepest/fs, NegSteepest, 
            TdiaRise/fs, DiaRise, SteepDiaRise, TSystoDiaRise/fs, TdiaToEnd/fs, Ratio]
    
    point_feat_name = point_feat_name + ['Ts_norm', 'Td_norm', 'Tsteepest_norm', 'TNegSteepest_norm',
                       'TdiaRise_norm', 'TSystoDiaRise_norm', 'TdiaToEnd_norm']
    point_feat = point_feat + [Ts/Tc, Td/Tc, Tsteepest/Tc, TNegSteepest/Tc, 
            TdiaRise/Tc, TSystoDiaRise/Tc, TdiaToEnd/Tc]
    
    width_names = []
    width_feats = []

    for per in [0.25,0.50,0.75]:
        SW, DW = width_at_per(per, cycle, peak, fs)
        per_str = str(int(per*100))
        width_names += ['SW'+per_str, 'SW'+per_str+'_norm', 
                               'DW'+per_str, 'DW'+per_str+'_norm', 
                               'SWaddDW'+per_str, 'SWaddDW'+per_str+'_norm',
                               'DWdivSW'+per_str]
        width_feats +=[SW/fs, SW/Tc,
                        DW/fs, DW/Tc,
                        (SW+DW)/fs, (SW+DW)/Tc,
                        DW/SW]
        
    point_feat_name += width_names
    point_feat += width_feats
    
    min_val=np.min(cycle)
    S1 = np.trapz(cycle[:Tsteepest]-min_val)
    S2 = np.trapz(cycle[Tsteepest:peak]-min_val)
    S3 = np.trapz(cycle[peak:TdiaRise]-min_val)
    S4 = np.trapz(cycle[TdiaRise:]-min_val)
    AUCsys = S1+S2
    AUCdia = S3+S4
    area_feat_name = ['S1','S2','S3','S4','AUCsys','AUCdia']
    area_feat = [S1,S2,S3,S4,AUCsys,AUCdia]
    
    area_feat_name += ['S1_norm','S2_norm','S3_norm','S4_norm','AUCsys_norm','AUCdia_norm']
    area_feat += [S1/AUCsys,S2/AUCsys,S3/AUCdia,S4/AUCdia,AUCsys/(AUCsys+AUCdia),AUCdia/(AUCsys+AUCdia)]
    
    SQI_skew = skew(cycle,0.3)
    SQI_kurtosis = kurtosis(cycle)
    sqi_feat_name = ['SQI_skew','SQI_kurtosis']
    sqi_feat = [SQI_skew,SQI_kurtosis]
    
    feat_name = point_feat_name + area_feat_name +sqi_feat_name
    feat = point_feat + area_feat + sqi_feat
    
    feats_header_apg, feats_apg = extract_apg_feat(cycle, vpg, peak, w, y, z, fs)
    
    feat_name += feats_header_apg
    feat += feats_apg
    
    return np.array(feat_name),np.array(feat)


def signal_fft(data, fs, norm='ortho'):
    org_fft = np.fft.fft(data, norm=norm)
    abs_org_fft = np.abs(org_fft)
    freq = np.fft.fftfreq(data.shape[0], 1/fs)
    abs_org_fft = abs_org_fft[freq > 0]
    freq = freq[freq > 0]
    
    return freq, abs_org_fft


def get_fft_peaks(fft, freq, fft_peak_distance = 28, num_iter = 5):
    peaks = scipy.signal.find_peaks(fft[:len(fft)//6], distance=fft_peak_distance)[0]  # all of observed distance > 28
    peaks = peaks[peaks>fft_peak_distance]
    peaks = peaks[0:num_iter]
    return peaks


def fft_peaks_neighbor_avg(fft, fft_peaks, fft_neighbor_avg_interval = 6):    
    fft_peaks_neighbor_avgs = []
    for peak in fft_peaks:
        start_idx = peak - fft_neighbor_avg_interval
        end_idx = peak + fft_neighbor_avg_interval
        fft_peaks_neighbor_avgs.append(fft[start_idx:end_idx].mean())
    return np.array(fft_peaks_neighbor_avgs)


def extract_cycles_all_ppgs(waveforms, ppg_peaks, hr_offset, match_position, remove_first = True):
    offset = np.round(hr_offset).astype("int") 

    waveforms_cycles = {
        "ppg_cycles": [],
        "vpg_cycles": [],
        "apg_cycles": [],
        "ppg3_cycles": [],
        "ppg4_cycles": []
    }

    if remove_first:
        ppg_peaks = ppg_peaks[1:-1]
    else:
        if ppg_peaks[0] == 0:
            ppg_peaks = ppg_peaks[1:]
        if ppg_peaks[-1] == len(waveforms['ppg'])-1:
            ppg_peaks = ppg_peaks[:-1]
            
    
    lower_offset = offset * 0.25
    upper_offset = offset * 0.75

    for p in ppg_peaks:

        start = np.round(p - lower_offset).astype("int")
        end = np.round(p + upper_offset).astype("int")

        if match_position == "dia_notches":
            tolerance = 0.1

            start = np.round(p - offset * (0.25 + tolerance)).astype("int")
            end = np.round(p + offset * (0.75 + tolerance)).astype("int")
            
            # check range
            if (start < 0) or (p <= start) or \
               (int(p + offset * (0.75-tolerance)) < 0) or (end <= int(p + offset * (0.75-tolerance))) or \
               (len(waveforms["ppg"][start:p])) == 0 or len(waveforms["ppg"][int(p + offset * (0.75-tolerance)):end]) <= 0:
                continue
            
            # stand and end from valleys
            start = start + np.argmin(waveforms["ppg"][start:p])
            end = int(p + offset * (0.75-tolerance)) + np.argmin(waveforms["ppg"][int(p + offset * (0.75-tolerance)):end])

        if (start < 0) or (end > waveforms["ppg"].shape[0]):
            continue

        for waveform_name in waveforms:
            waveforms_cycles[waveform_name + "_cycles"].append(waveforms[waveform_name][start:end])

    for waveform_name in waveforms:
        waveforms_cycles[waveform_name + "_cycles"] = np.array(waveforms_cycles[waveform_name + "_cycles"], dtype=object)
        
    return waveforms_cycles


def mean_norm_cycles(cycles, resample_length = 80):
    
    normalized_cycles = []
    for cycle in cycles:
        normalized_cycle = scipy.signal.resample(cycle, resample_length)
        normalized_cycle = waveform_norm(normalized_cycle)
        normalized_cycles.append(normalized_cycle)

    if len(normalized_cycles) > 0:
        normalized_cycles = np.array(normalized_cycles)
        
    avg_normalized_cycles = np.median(normalized_cycles, axis=0)
    return avg_normalized_cycles, normalized_cycles


def max_neighbor_mean(mean_cycles, neighbor_mean_size = 5):
    ppg_start_idx = max(np.argmax(mean_cycles) - neighbor_mean_size, 0)
    ppg_end_idx = min(np.argmax(mean_cycles) + neighbor_mean_size, len(mean_cycles))

    if ppg_start_idx == ppg_end_idx: 
        ppg_end_idx += 1

    return mean_cycles[ppg_start_idx:ppg_end_idx].mean()


def histogram_up_down(mean_cycles, num_up_bins, num_down_bins, ppg_max_idx):
    H_up, bins_up = np.histogram(mean_cycles[:ppg_max_idx], bins=num_up_bins, range=(0,1), density=True)
    H_down, bins_down = np.histogram(mean_cycles[ppg_max_idx:], bins=num_down_bins, range=(0,1), density=True)

    return H_up, H_down, bins_up, bins_down


def USDC(cycles, USDC_resample_length):    
    usdc_features = []
    for cycle in cycles:
        max_idx = np.argmax(cycle)
        cycle = scipy.signal.resample(cycle[:max_idx+1], USDC_resample_length)
        max_idx = len(cycle) - 1
        usdc = (cycle * (cycle[max_idx] - cycle[0]) * np.arange(len(cycle)) - cycle * max_idx + cycle[0] * max_idx) / (np.sqrt((cycle[max_idx] - cycle[0]) ** 2 + max_idx ** 2))

        interval = 3
        usdc_feature = []
        for idx in range(0, max_idx, interval):
            if idx+interval < len(usdc):
                usdc_feature.append(usdc[idx:idx+interval].mean())

        usdc_features.append(usdc_feature)

    usdc_features = np.array(usdc_features)
    mean_usdc_features = usdc_features.mean(axis=0)

    return mean_usdc_features


def DSDC(cycles, DSDC_resample_length):
    dsdc_features = []
    for cycle in cycles:

        max_idx = np.argmax(cycle)
        cycle = scipy.signal.resample(cycle[max_idx:], DSDC_resample_length)
        l = len(cycle) - 1
        max_idx = 0
        dsdc = (cycle * (cycle[l] - cycle[max_idx]) * np.arange(len(cycle)) - (l - max_idx) * cycle + cycle[max_idx] * l - cycle[l] * max_idx) / np.sqrt((cycle[l] - cycle[max_idx]) ** 2 + (l - max_idx) ** 2)

        interval = 3
        dsdc_feature = []
        for idx in range(max_idx, len(dsdc), interval):
            if idx+interval < len(dsdc):
                dsdc_feature.append(dsdc[idx:idx+interval].mean())

        dsdc_features.append(dsdc_feature)

    dsdc_features = np.array(dsdc_features)
    mean_udsc_features = dsdc_features.mean(axis=0)

    return mean_udsc_features


class PPG:
    def __init__(self, data, fs, cycle_size=128):
        self.data = data
        self.idata = data.max() - data
        self.fs = fs
        self.cycle_size = cycle_size

    def peaks(self, **kwargs): 
        return find_peaks(self.data, scale=int(self.fs))

    def vpg(self, **kwargs):
        vpg = self.data[1:] - self.data[:-1]
        padding = np.zeros(shape=(1))
        vpg = np.concatenate([padding, vpg], axis=-1)
        return vpg

    def apg(self, **kwargs):
        apg = self.data[1:] - self.data[:-1]  # 1st Derivative
        apg = apg[1:] - apg[:-1]  # 2nd Derivative

        padding = np.zeros(shape=(2))
        apg = np.concatenate([padding, apg], axis=-1)
        return apg
    
    def ppg3(self, **kwargs):
        ppg3 = self.data[1:] - self.data[:-1]  # 1st Derivative
        ppg3 = ppg3[1:] - ppg3[:-1]  # 2nd Derivative
        ppg3 = ppg3[1:] - ppg3[:-1]  # 3nd Derivative

        padding = np.zeros(shape=(3))
        ppg3 = np.concatenate([padding, ppg3], axis=-1)
        return ppg3
    
    def ppg4(self, **kwargs):
        ppg4 = self.data[1:] - self.data[:-1]  # 1st Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 2nd Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 3nd Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 4nd Derivative

        padding = np.zeros(shape=(4))
        ppg4 = np.concatenate([padding, ppg4], axis=-1)
        return ppg4

    def hr(self, **kwargs):
        try: return self.fs / np.median(np.diff(self.peaks())) * 60
        except: return 0

    def diastolic_notches(self, **kwargs):
        notches = find_peaks(-self.data, scale=int(self.fs))        
        return notches    
       
    def features_extractor(self, filtered=False, 
        fft_peak_distance = 28, fft_neighbor_avg_interval = 6, 
        resample_length = 80,
        neighbor_mean_size = 5,
        num_up_bins = 5,
        num_down_bins = 10, 
        remove_first = True,
        one_cycle_sig = False):
        if one_cycle_sig:
            remove_first = False
        
        features = {}
        
        USDC_resample_length = resample_length // 4
        DSDC_resample_length = resample_length // 4 * 3

        ppg = self.data
        ppg = waveform_norm(ppg)
        if filtered:
            vpg = mean_filter_normalize( self.vpg(), int(self.fs), 0.75, 10, 1)
            apg = mean_filter_normalize( self.apg(), int(self.fs), 0.75, 10, 1)
            ppg3 = mean_filter_normalize( self.ppg3(), int(self.fs), 0.75, 10, 1)
            ppg4 = mean_filter_normalize( self.ppg4(), int(self.fs), 0.75, 10, 1)
        else:
            vpg = waveform_norm(self.vpg())
            apg = waveform_norm(self.apg())
            ppg3 = waveform_norm(self.ppg3())
            ppg4 = waveform_norm(self.ppg4())
            
        sys_peaks = find_peaks(self.data, scale=int(self.fs))
        dia_notches = self.diastolic_notches()
        
        if one_cycle_sig:
            sys_peaks = np.array([self.data.argmax()])
            dia_notches = np.array([0,len(self.data)-1])
        
        waveforms = {
            "ppg": ppg,
            "vpg": vpg,
            "apg": apg,
            "ppg3": ppg3,
            "ppg4": ppg4
        }
        
        for waveform_name in waveforms:
            freq, fft = signal_fft(waveforms[waveform_name], self.fs)
            fft = fft / np.linalg.norm(fft)
            fft_peaks = get_fft_peaks(fft, freq, fft_peak_distance, num_iter=5)
            fft_peaks_neighbor_avgs = fft_peaks_neighbor_avg(fft, fft_peaks, fft_neighbor_avg_interval)
            features[waveform_name + "_fft_peaks"] = fft_peaks
            features[waveform_name + "_fft_peaks_heights"] = fft[fft_peaks]
            features[waveform_name + "_fft_peaks_neighbor_avgs"] = fft_peaks_neighbor_avgs
        
        p2p = np.median(np.diff(sys_peaks))
        if one_cycle_sig:
            p2p = len(self.data)
        
        waveforms_cycles_match_peak = extract_cycles_all_ppgs(waveforms, sys_peaks, p2p, "sys_peak", remove_first)
        waveforms_cycles_match_valleys = extract_cycles_all_ppgs(waveforms, sys_peaks, p2p, "dia_notches", remove_first)
        
        if one_cycle_sig:
            waveforms_cycles_match_valleys = {
                "ppg_cycles": [],
                "vpg_cycles": [],
                "apg_cycles": [],
                "ppg3_cycles": [],
                "ppg4_cycles": []
            }
            for waveform_name in waveforms:
                waveforms_cycles_match_valleys[waveform_name + "_cycles"] = np.array([waveforms[waveform_name]], dtype=object)
            
            waveforms_cycles_match_peak = copy.deepcopy(waveforms_cycles_match_valleys)
            
        
        if len(waveforms_cycles_match_peak["ppg_cycles"]) == 0:
            raise RuntimeError("Feature extractor warning: There are no cycles")
        if len(waveforms_cycles_match_valleys["ppg_cycles"]) == 0:
            raise RuntimeError("Feature extractor warning: There are no cycles 2")
        
        features["hr"] = self.fs / p2p * 60 
        features["p2p"] = p2p

        for waveform_name in waveforms:
            cycles = waveforms_cycles_match_peak[waveform_name + "_cycles"]
            mean_cycles, norm_cycles = mean_norm_cycles(cycles, resample_length)
            if "ppg" == waveform_name:
                features["ppg_mean_cycles_match_peak"] = mean_cycles
            neighbor_mean = max_neighbor_mean(mean_cycles, neighbor_mean_size)
            features[waveform_name + "_max_neighbor_mean"] = neighbor_mean
            features[waveform_name + "_min"] = np.argmin(mean_cycles)
            
        ppg_cycles_match_valleys = waveforms_cycles_match_valleys["ppg_cycles"]
        ppg_mean_cycles_match_valleys, ppg_norm_cycles_match_valleys = mean_norm_cycles(ppg_cycles_match_valleys, resample_length)
        ppg_max_idx = np.argmax(ppg_mean_cycles_match_valleys)
        
        for waveform_name in waveforms:
            cycles = waveforms_cycles_match_valleys[waveform_name + "_cycles"]
            mean_cycles, norm_cycles = mean_norm_cycles(cycles, resample_length)
            if "ppg" == waveform_name:
                features["ppg_mean_cycles_match_valleys"] = mean_cycles
            H_up, H_down, bins_up, bins_down = histogram_up_down(mean_cycles, num_up_bins, num_down_bins, ppg_max_idx)
            features[waveform_name + "_histogram_up"] = H_up
            features[waveform_name + "_histogram_down"] = H_down
            features[waveform_name + "_max"] = np.argmax(mean_cycles)
        
        usdc = USDC(ppg_norm_cycles_match_valleys, USDC_resample_length)
        dsdc = DSDC(ppg_norm_cycles_match_valleys, DSDC_resample_length)
        features["usdc"] = usdc
        features["dsdc"] = dsdc
                
        return features


if __name__ == "__main__":
    pass
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt 
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import sys
sys.path.append(".")
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error
from scipy.stats import skew, kurtosis
from configs.seed import *


class Pulse:
    def __init__(self, data, pressure, kind):
        self.data = data
        self.pressure = pressure
        self.kind = kind


class PPGCluster:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None

    def _extract_features(self, segment):
        peaks, _ = find_peaks(segment)
        
        n_peaks = len(peaks)
        idx_first_peak = peaks[0] if n_peaks > 0 else 0
        idx_second_peak = peaks[1] if n_peaks > 1 else 0
        idx_notch = np.argmin(segment[idx_first_peak:idx_second_peak]) + idx_first_peak if n_peaks > 1 else 0
        notch_amplitude = segment[idx_notch]
        amplitude_first_peak = segment[idx_first_peak]
        amplitude_second_peak = segment[idx_second_peak] if n_peaks > 1 else 0
        notch_ratio = notch_amplitude / amplitude_first_peak if amplitude_first_peak != 0 else 0
        second_peak_ratio = amplitude_second_peak / amplitude_first_peak if amplitude_first_peak != 0 else 0
        peak_to_peak_interval = idx_second_peak - idx_first_peak if n_peaks > 1 else 0
        
        # Statistical features
        mean_val = np.mean(segment)
        std_val = np.std(segment)
        var_val = np.var(segment)
        skewness_val = skew(segment)
        kurtosis_val = kurtosis(segment)
        
        features = [
            n_peaks, idx_first_peak, idx_second_peak, idx_notch, notch_ratio, second_peak_ratio, 
            peak_to_peak_interval, mean_val, std_val, var_val, skewness_val, kurtosis_val
        ]
        
        return features
    
    def _compute_distance_matrix(self, X):
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i > j:
                    dist = mean_squared_error(X[i], X[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        return distance_matrix

    def fit_predict(self, segments):
        features = np.array([self._extract_features(segment) for segment in segments])        
        features = MinMaxScaler().fit_transform(features)
        distance_matrix = self._compute_distance_matrix(features)
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        labels = self.model.fit_predict(distance_matrix)
        return labels


class PPGSpectralClustering:
    def __init__(self, n_clusters=6, affinity='cosine'):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    
    def _compute_similarity_matrix(self, X):
        n_samples = X.shape[0]
        similarity_matrix = 0
        if self.affinity == 'rbf':
            distances = pairwise_distances(X, metric='euclidean')
            similarity_matrix = np.exp(-distances)
        
        elif self.affinity == 'cosine':
            similarity_matrix = cosine_similarity(X)
            similarity_matrix = (similarity_matrix + 1) / 2.0
            similarity_matrix += 1e-5

        return similarity_matrix
    
    def fit_predict(self, X):
        X = MinMaxScaler().fit_transform(X)
        X = self._compute_similarity_matrix(X)        
        labels = self.model.fit_predict(X)
        return labels
    

class PPGKMeans:
    def __init__(self, n_clusters=3, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    def fit_predict(self, segments):
        cosine_features = cosine_similarity(segments)
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = self.model.fit_predict(cosine_features)
        return labels


def plot_tsne(ppg_data, labels):
    tsne = TSNE(n_components=2, random_state=42)
    ppg_tsne = tsne.fit_transform(ppg_data)
    plt.figure(figsize=(10, 8))
    plt.scatter(ppg_tsne[:, 0], ppg_tsne[:, 1], c=labels, edgecolors='k', cmap=plt.cm.Paired)
    plt.colorbar()
    plt.title("t-SNE Visualization with Clustering Results")
    plt.show()
    ''' 3D Vizualization
    tsne = TSNE(n_components=3, random_state=42)
    ppg_tsne = tsne.fit_transform(ppg_data)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(ppg_tsne[:, 0], ppg_tsne[:, 1], ppg_tsne[:, 2], c=labels, edgecolors='k', cmap=plt.cm.Paired)
    plt.colorbar(scatter)
    ax.set_title("3D t-SNE Visualization with Clustering Results")
    plt.show()
    '''


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, fs, cutoff, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def peakfind(x):
    peaks, _ = scipy.signal.find_peaks(x, distance=40, prominence=200)
    return peaks


def deriv_n(signal):
    r = np.gradient(signal, 1)
    while len(r) < len(signal):
        r = np.append(r, r[-1])
    return r


def classify(sig):
    sig = sig/max(sig)
    sig = signal.resample(sig, 100)
    
    if min(sig) < -0.3:
        return '3'
    
    # Identify obvious peaks above a threshold
    peaks, _ = signal.find_peaks(sig, prominence=0.02)
    peaks = [y for y in peaks if sig[y] >= 0.45]
    
    # If the are many or no obvious peaks, reject
    if len(peaks) >= 3 or len(peaks) == 0:
        return '3'
    
    if len(peaks) == 2:
        
        # If the 2 peaks are too far, abnormal
        if peaks[1] - peaks[0] > 65:
            return '3'
        
        # If the first peak is significantly larger than the second
        if (sig[peaks[0]] - sig[peaks[1]]) > 0.1:
            # Abnormal peak placement
            if peaks[1] - peaks[0] < 20 or peaks[0] > 25:
                return '3'
            
            # Dicrotic notch detection
            # If there is a peak in the inverted signal between the 2 detected peaks
            # And it is not too far down or up
            p2, _ = signal.find_peaks(-sig)
            notch_peaks = [p for p in p2 \
                           if p < peaks[1] \
                           and p > peaks[0] \
                           and sig[p] < sig[peaks[1]]]
            
            # Exclude the edges, because they are typically not important and may have small bumps
            ps, _ = signal.find_peaks(sig[10:85])
            ps += 10
            nps, _ = signal.find_peaks(-sig[10:85])
            nps += 10
            if len(notch_peaks) == 1: # 2 Clear peaks + notch

                # Check if signal is too unreliable (Many peaks)
                # or peaks within the sensitive area between 
                # systolic and diastolic peak
                if len(ps) >= 3 \
                    or len(nps) >= 2 \
                    or len([p for p in ps if p < peaks[1] + 15 and p > peaks[0] - 10]) > 2 \
                    or len([np for np in nps if np < peaks[1] + 15 and np > peaks[1] - 10]) > 1:
                    
                    return '3'
                
                # If the dicrotic notch is very off-centre or extremely low
                d_notch = notch_peaks[0]
                if d_notch > (0.4 * peaks[1] - peaks[0]) + peaks [0] and \
                    sig[d_notch] > sig[peaks[1]] - 30:
                    return '2L'
                else:
                    return '3'
            else:
                # If the signal is too unstable
                if len(ps) >= 3:
                    return '3'
                return '1L' # No clear notch

        # If the peaks are nearly equal
        else:
            ps, _ = signal.find_peaks(sig[:80])
            nps, _ = signal.find_peaks(-sig[:80])
            
            # Identify the notch between the two peaks
            notch_peaks = [p for p in nps \
               if p < peaks[1] \
               and p > peaks[0] \
               and sig[p] < sig[peaks[1]]]

            # If there are many peaks, more than 1 notch, or the notch is lower than normal, reject    
            if len(ps) >= 3 \
                or len(nps) >= 2 \
                or len(notch_peaks) > 1 \
                or len(notch_peaks) == 1 and sig[notch_peaks[0]] < min(sig[peaks[0]], sig[peaks[1]]) - 0.15:
                return '3'
            return '2E'
            
    else:
        # At this point, there should only be 1 peak in the whole wave
        peaks, _ = signal.find_peaks(sig, prominence=0.02)
        if len(peaks) > 1:
            return '3'
        
        # If the peak is far to the left
        if peaks[0] <= 25:
            ps, _ = signal.find_peaks(sig)
            nps, _ = signal.find_peaks(-sig)
            if len(ps) > 2 or len(nps) >= 2: # If there are too many peaks in the wave, reject
                    return '3'
            return '1L'
        
        return '1'


def load_from_npy(path_to_npy_file='assets/train.npy'):
    segments = np.load(path_to_npy_file)
    labels = []
    src_segments, ref_segments, pressure_segments = segments[:,0,:,:].reshape(segments.shape[0],-1), segments[:,1,:,:].reshape(segments.shape[0],-1), segments[:,2,:,:].reshape(segments.shape[0],-1)
    for segment in src_segments:
        labels.append(classify(segment))
    return src_segments, ref_segments, labels


def plot_pair_from_npy(file_name="assets/train.npy"):
    '''
    A 2x5 figure to show different waveforms of the PPG, 
    where the first row plots for the source signal, 
    the second row for the ref signal. The order in each row is 2L, 1, 1L, 2E, 3.
    '''
    order = ['2L', '1', '1L', '2E', '3']
    segments = np.load(file_name)
    src_segments, ref_segments, _ = segments[:,0,:,:], segments[:,1,:,:], segments[:,2,:,:] 
    class_track = dict()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    idx = 0
    random.seed(111)
    while len(class_track) < 5:
        i = random.choice(range(len(src_segments)))
        ppg_type = classify(src_segments[i].reshape(-1))
        if ppg_type in class_track:
            continue
        class_track[ppg_type] = i
        print(ppg_type, i)

    for type in order:
        axes[0, idx].plot(src_segments[class_track[type]].reshape(-1))
        axes[0, idx].set_title(f'{type}')
        axes[1, idx].plot(ref_segments[class_track[type]].reshape(-1))
        idx+=1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_pair_from_npy()    



    
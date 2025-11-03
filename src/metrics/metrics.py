import numpy as np
import torch
from scipy.stats import skew, kurtosis
from src.utils.preprocess import get_peaks_info_of_segments, cycle_helper, get_feet
from src.utils.feature import extract_feat_cycle
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from tslearn.metrics import dtw, dtw_path


class SignalComparison:
    def __init__(self, cfgs=None):
        self.cfgs = cfgs

    def cosine_similarity(self, original, reference):
        dot_product = torch.sum(original * reference, dim=-1)        
        original = torch.norm(original, dim=-1)
        reference = torch.norm(reference, dim=-1)        
        cosine_similarity = dot_product / (original * reference)
        return cosine_similarity

    def dtw_similarity(self, original, reference):
        similarities = []
        batch_size = original.shape[0]
        
        for i in range(batch_size):
            distance = dtw(original[i], reference[i])
            similarities.append(distance)
        
        return np.mean(similarities)

    def pearson_cc(self, original, reference):
        correlation_coefficients = []
        for rec, ref in zip(original, reference):
            corr_coeff, _ = pearsonr(rec, ref)
            correlation_coefficients.append(corr_coeff)
        
        mean_correlation = np.mean(correlation_coefficients)
        
        return mean_correlation     

    def skewness_errors(self, original, reference):
        total_err = 0
        valid = 0
        for i in range(original.shape[0]):
            src_cycles, ref_cycles, _ = cycle_helper(original[i], reference[i], return_cycles=True)
            for (src_cycle, ref_cycle) in zip(src_cycles, ref_cycles):
                src_skew = skew(src_cycle)
                ref_skew = skew(ref_cycle)
                if np.isnan(src_skew) or np.isnan(ref_skew) or ref_skew == 0:
                    continue
                total_err += (np.abs((-src_skew + ref_skew) / ref_skew))
                valid+=1

        return total_err / valid
    
    def kurtosis_errors(self, original, reference):
        total_err = 0
        valid = 0
        for i in range(original.shape[0]):
            src_cycles, ref_cycles, _ = cycle_helper(original[i], reference[i], return_cycles=True)
            for (src_cycle, ref_cycle) in zip(src_cycles, ref_cycles):
                src_skew = kurtosis(src_cycle)
                ref_skew = kurtosis(ref_cycle)
                if np.isnan(src_skew) or np.isnan(ref_skew) or ref_skew == 0:
                    continue
                total_err += (np.abs((-src_skew + ref_skew) / ref_skew))
                valid+=1

        return total_err / valid

    def peak_errors(self, original, reference):
        total_err = np.array([0.0,0.0,0.0])
        for i in range(original.shape[0]):
            peaks_src, peaks_ref = get_peaks_info_of_segments(original[i], reference[i], return_ids=False, return_ratio=True, distance=10)
            total_err += np.mean(np.abs((-peaks_src + peaks_ref))/peaks_ref, axis=0)
        return total_err / (original.shape[0])

    def index_errors(self, original, reference):
        total_err = np.array([0.0,0.0,0.0])
        for i in range(original.shape[0]):
            ids_src, ids_ref = get_peaks_info_of_segments(original[i], reference[i], return_ids=True, distance=10)
            total_err += np.mean((np.abs(-ids_src + ids_ref)/ids_ref), axis=0)
        return total_err / (original.shape[0])
    
    def feat_errors(self, original, reference):
        sys_err = np.array([0.0] * 3)
        area_err = np.array([0.0] * 2)
        width_at_per_err = np.array([0.0] * 6)
        dia_err = np.array([0.0] * 5)
        sqi_err = np.array([0.0] * 2)
        
        contain_dia = 0

        valid_src_dia = 0
        valid_ref_dia = 0

        total_src_cycles = 0
        total_ref_cycles = 0
        
        sqi_raw = []
        sqi_ref = []

        for i in range(original.shape[0]):
            breakdown_seg_in, breakdown_seg_ref, _ = cycle_helper(original[i].copy().reshape(-1), reference[i].copy().reshape(-1), return_cycles=True)
            ###################### IN SEGMENTS #######################

            feat_name, src_feats, dia_src_feat_name, dia_src_feats, src_valid = extract_feat_cycle(breakdown_seg_in[1:-1], fs=128)
            valid_src_dia += src_valid
            src_feats = dict(zip(feat_name, src_feats))
            dia_src_feats = dict(zip(dia_src_feat_name, dia_src_feats))
            total_src_cycles += (len(breakdown_seg_in)-2)
            
            sys_src_feats = np.array([src_feats["sys_amp"], src_feats["Ts"], src_feats["Td"]] )
            sqi_src_feats = np.array([src_feats["skewness"], src_feats["kurtosis"]] )
            sqi_raw.append(sqi_src_feats)

            width_at_per_src_feats = np.array([src_feats["SW75_norm"], src_feats["SW50_norm"], src_feats["SW25_norm"], src_feats["DW75_norm"], src_feats["DW50_norm"], src_feats["DW25_norm"]])
            area_src_feats = np.array([src_feats["AUCsys"],src_feats["AUCdia"]])
            if len(dia_src_feats) > 0:
                dia_src_feats = np.array([dia_src_feats["notch_amp"],dia_src_feats["notch_idx"],dia_src_feats["dia_amp"],dia_src_feats["dia_idx"],dia_src_feats["p2p_time"]])
            else:
                dia_src_feats = np.array([0,0,0,0,0])

            ####################### REF SEGMENTS ######################
            feat_name, ref_feats, dia_ref_feat_name, dia_ref_feats, ref_valid = extract_feat_cycle(breakdown_seg_ref[1:-1], fs=128)
            ref_feats = dict(zip(feat_name, ref_feats))
            dia_ref_feats = dict(zip(dia_ref_feat_name, dia_ref_feats))
            total_ref_cycles += (len(breakdown_seg_ref)-2)
            valid_ref_dia += ref_valid

            sys_ref_feats = np.array([ref_feats["sys_amp"], ref_feats["Ts"], ref_feats["Td"]] )
            sqi_ref_feats = np.array([ref_feats["skewness"], ref_feats["kurtosis"]] )
            sqi_ref.append(sqi_ref_feats)
            

            width_at_per_ref_feats = np.array([ref_feats["SW75_norm"], ref_feats["SW50_norm"], ref_feats["SW25_norm"], ref_feats["DW75_norm"], ref_feats["DW50_norm"], ref_feats["DW25_norm"]])
            area_ref_feats = np.array([ref_feats["AUCsys"],ref_feats["AUCdia"]]) 
            dia_ref_feats =  np.array([dia_ref_feats["notch_amp"],dia_ref_feats["notch_idx"],dia_ref_feats["dia_amp"],dia_ref_feats["dia_idx"],dia_ref_feats["p2p_time"]])
            ###########################################################

            sys_err += np.abs((-sys_src_feats + sys_ref_feats) / sys_ref_feats)
            area_err += np.abs((-area_src_feats + area_ref_feats) / area_ref_feats)
            width_at_per_err += np.abs((-width_at_per_src_feats + width_at_per_ref_feats) / width_at_per_ref_feats)
            dia_err += np.abs((-dia_src_feats + dia_ref_feats)/ dia_ref_feats)
            sqi_err += np.abs((-sqi_src_feats + sqi_ref_feats))

        return sys_err / original.shape[0], area_err / original.shape[0], sqi_err / original.shape[0],  width_at_per_err / original.shape[0], dia_err/original.shape[0], valid_src_dia/total_src_cycles, valid_ref_dia/total_ref_cycles
        
    def compare(self, original, reference):
        batch_size = original.shape[0]
        mse = torch.nn.MSELoss()(original, reference).detach().cpu().numpy()
        mae = torch.nn.L1Loss()(original, reference).detach().cpu().numpy()
        
        original = original.cpu().detach().numpy().reshape(batch_size, -1)
        reference = reference.cpu().detach().numpy().reshape(batch_size, -1)
        
        dtw = self.dtw_similarity(original, reference)
        pcc = self.pearson_cc(original, reference)
        
        sys_errs, area_errors, sqi_errs, width_at_per_errs, dia_errs, valid_src_percentage, valid_ref_percentage = self.feat_errors(original, reference)
        return mae, mse, dtw, dtw/8, pcc, sys_errs[0], sys_errs[1], sys_errs[2], dia_errs[0], dia_errs[1], dia_errs[2], dia_errs[3], dia_errs[4], valid_src_percentage, area_errors[0], area_errors[1], sqi_errs[0], sqi_errs[1], width_at_per_errs[0], width_at_per_errs[1], width_at_per_errs[2], width_at_per_errs[3], width_at_per_errs[4], width_at_per_errs[5]
    
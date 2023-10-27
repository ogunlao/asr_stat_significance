import random
import numpy as np
from scipy.stats import bootstrap

class StatisticalSignificance:
    """Performs statistical test between two ASR models."""
    def __init__(self, file_path, sep=",", total_batch=1000, use_gaussian_appr=False,):
        """
        Args:
            file_path (str): Path to the wer info for each test sentence
            sep (str): Separator used in file_a and file_b for error and total number of words. (default is ",")
            total_batch (int): Total amount of bootstrap sampling runs. Typical values are 10^2, 10^3, 10^4. (default is 1000)
            use_gaussian_appr (bool): Either to manually compute empirical percentiles or use gaussian approximation. (default is False)
        """
        
        self.file_path = file_path
        self.total_batch = total_batch
        
        self.data_wer = self.process_text_file(file_path, sep=sep)
        self.z_scores = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576,
        }
        self.use_gaussian_appr = use_gaussian_appr
    
    def process_text_file(self, file_path, sep):
        data_wer = {}
        with open(file_path, "r+") as f:
            for line in f:
                block_data = line.split(sep)
                if len(block_data) == 3:
                    # all data assumed to belong to only one block
                    edit_wer_a, edit_wer_b, num_words = block_data
                    block = 0 # default block
                elif len(block_data) == 4:
                    # has information on blocks
                    edit_wer_a, edit_wer_b, num_words, block = block_data
                    
                if block in data_wer:
                    data_wer[block].append((int(edit_wer_a), int(edit_wer_b), int(num_words)))
                else:
                    data_wer[block] = [(int(edit_wer_a), int(edit_wer_b), int(num_words))]
        
        for block in data_wer:
            data_wer[block]  = np.array(data_wer[block])
        
        return data_wer
        
    def random_sample(self, data, num_samples,):
        """ Randomly samples from data with replacement.
        """
        
        random_index = np.random.randint(0, data.shape[0], size=num_samples)
        return data[random_index]
    
    def wer_change(self, data):
        return np.sum(data[:, 1] - data[:, 0]) / np.sum(data[:, 2])
    
    def standard_error(self, wer_change, wer_change_mean):
        std_dev = np.sum((wer_change - wer_change_mean)**2)/(len(wer_change)-1)
        return np.sqrt(std_dev)
    
    def bootstap_sampling(self, data, num_samples_per_batch):
        change_in_wer_arr = []
        for _ in range(self.total_batch):
            # sample a batch from the entire data
            sample_data = self.random_sample(data, num_samples=num_samples_per_batch,)
            
            # compute batch wer diff
            change_in_wer_batch = self.wer_change(sample_data)
            change_in_wer_arr.append(change_in_wer_batch)
            
        return np.array(change_in_wer_arr)
    
    def bootstap_sampling_block(self, data, num_samples_per_block):
        change_in_wer_arr = []
        for _ in range(self.total_batch):
            sample_data = []
            for block in data:
                block_sample_data = self.random_sample(data[block], num_samples=num_samples_per_block,)
                sample_data.append(block_sample_data)
            sample_data = np.array(sample_data)
            
            # compute batch wer diff
            change_in_wer_batch = self.wer_change(sample_data)
            change_in_wer_arr.append(change_in_wer_batch)
            
        return np.array(change_in_wer_arr)
    
    def compute_significance(self, num_samples_per_batch, ci,  use_blockwise_bootstrap=False,):
        """
        Args:
            num_samples_per_batch (int): The number of WER/CER samples selected from the files per model \
                (default is 1000)
            ci (float): Confidence Interval to be used for computation. \
                Typical CI include 90%, 95% and 99% (default is 0.95)
            use_blockwise_bootstrap (bool): Perform bootstrap sampling based on blocks e.g. speakers. (default is False)
        """
        
        assert ci < 1.0, f"Sorry ci has to be less than 1.0 . Given ci = {ci}"
        if self.use_gaussian_appr:
            assert ci in self.z_scores, "Sorry, only confidence intervals of 0.90, 0.95 or 0.99 are supported if `self.use_gaussian_appr=True`"
            z_score = float(self.z_scores[ci])
        
        data = self.data_wer
        num_samples_per_block = num_samples_per_batch//len(data)
        absolute_wer_diff = None
        if use_blockwise_bootstrap:
            change_in_wer_arr = self.bootstap_sampling_block(data, num_samples_per_block)
        else:
            data_expanded = [data[block_data] for block_data in data]
            data_expanded = np.vstack(data_expanded) if len(data) > 1 else data_expanded[0]
            change_in_wer_arr = self.bootstap_sampling(data_expanded, num_samples_per_batch)
            absolute_wer_diff = self.wer_change(data_expanded)
        
        # compute standard_error
        wer_diff_bootstrap = np.mean(change_in_wer_arr)
        std_err_bootstrap = self.standard_error(change_in_wer_arr, 
                                           wer_change_mean=wer_diff_bootstrap)
        
        # compute ci intervals
        if self.use_gaussian_appr:
            ci_low, ci_high = wer_diff_bootstrap + z_score*std_err_bootstrap, wer_diff_bootstrap - z_score*std_err_bootstrap
        else:
            interval = (1.0 - ci)/2
            ci_low, ci_high = np.percentile(change_in_wer_arr, [(1.0-interval)*100], interval*100)
        
        return WER_DiffCI(wer_diff_bootstrap, ci_low, ci_high, std_err_bootstrap, absolute_wer_diff)
    
class WER_DiffCI:
    def __init__(self, wer_diff_bootstrap, ci_high, ci_low, std_err, wer_diff_absolute=None,):
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.std_err = std_err
        self.wer_diff_bootstrap = wer_diff_bootstrap
        self.wer_diff_absolute = wer_diff_absolute
    
    def is_significant(self):
        return (self.ci_low < 0) and (self.ci_high < 0)
    
    def __repr__(self):
        return f"WER_DiffCI(wer_diff_bootstrap={self.wer_diff_bootstrap}, ci_low={self.ci_low}, ci_high={self.ci_high}, std_err={self.std_err})"
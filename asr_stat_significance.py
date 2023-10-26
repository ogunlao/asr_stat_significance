import random
import numpy as np
from scipy.stats import bootstrap


class StatisticalSignificance:
    """Performs statistical test between two ASR models."""
    def __init__(self, file_a, file_b, sep=",", total_batch=1000,):
        """
        Args:
            file_a (str): Path to the files for ASR model A.
            file_a (str): Path to the files for ASR model B.
            sep (str): Separator used in file_a and file_b for error and total number of words. (default is ",")
            total_batch (int): Total amount of bootstrap sampling runs. Typical values are 10^2, 10^3, 10^4. (default is 1000)
        """
        
        self.file_a = file_a
        self.file_b = file_b
        self.total_batch = total_batch
        
        self.data_wer = [] # [edit_a, edit_b, total_num]
        self.z_scores = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576,
        }
        with open(file_a, "r+") as f:
            for line in f:
                edit_wer, num_words = line.split(sep)
                self.data_wer.append([int(edit_wer), int(num_words)])
        
        with open(file_b, "r+") as f:
            i = 0
            for line in f:
                edit_wer, num_words = line.split("|")
                
                current_data_wer = self.data_wer[i]
                assert int(num_words) == current_data_wer[1] # ensures that the same sentence is compared with model A
                current_data_wer = [current_data_wer[0], int(edit_wer), current_data_wer[1]]
                
                self.data_wer[i] = current_data_wer
                i+=1
                
        self.data_wer = np.array(self.data_wer)
        self.absolute_wer_diff = self.wer_change(self.data_wer)
        
                
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
    
    def compute_significance(self, data, num_samples_per_batch, ci):
        """
        Args:
            num_samples_per_batch (int): The number of WER/CER samples selected from the files per model \
                (default is 1000)
            ci (float): Confidence Interval to be used for computation. \
                Typical CI include 90%, 95% and 99% (default is 0.95)
        """
        
        assert ci < 1.0, f"Sorry ci has to be less than 1.0 . Given ci = {ci}"
        
        change_in_wer_arr = self.bootstap_sampling(data, num_samples_per_batch)
        
        # compute standard_error
        wer_diff_bootstrap = np.mean(change_in_wer_arr)
        se_bootstrap = self.standard_error(change_in_wer_arr, 
                                           wer_change_mean=wer_diff_bootstrap)
        
        # compute ci intervals
        interval = (1.0 - ci)/2
        ci_high, ci_low = np.percentile(change_in_wer_arr, [interval*100, (1.0-interval)*100])
        
        return WER_DiffCI(wer_diff_bootstrap, ci_low, ci_high, se_bootstrap, self.absolute_wer_diff)
                        
    
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
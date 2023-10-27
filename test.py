### This test compares the implemented class with traditional bootsrap method from scipy. ###

import random
import numpy as np
from scipy.stats import bootstrap

from asr_stat_significance import StatisticalSignificance

def func(*x, axis=0):
    x = x[0]

    axis = len(x.shape) - 2
    a = np.sum(x[1] - x[0], axis=axis)
    sum_a = np.sum(x[2], axis=axis)

    return a/sum_a


if __name__ == "__main__":
    np.random.seed(42)
    
    confidence_interval = 0.95
    num_samples_per_batch = 40 # samples will be selected with replacement
    
    si_obj = StatisticalSignificance(
        file_path="test_files/wer_file.txt", 
        total_batch=10000,
        sep="|",
        use_gaussian_appr=True
    )
    
    ci_obj  = si_obj.compute_significance(num_samples_per_batch=40, ci=confidence_interval)
    print(ci_obj)
    
    # compare to scipy bootstrap method
    data = si_obj.data_wer[0]
    res = bootstrap((data,), func, confidence_level=confidence_interval, 
                    method="bca", vectorized=True, batch=10000)
    print(res)
    
    assert np.sign(ci_obj.ci_low) == np.sign(res.confidence_interval.low)
    assert np.sign(ci_obj.ci_high) == np.sign(res.confidence_interval.high)
    
    print("Model Y is not significant:", ci_obj.is_significant() == False)
    
    print("Test passed !!!")
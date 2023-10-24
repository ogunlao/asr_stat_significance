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
        file_a="test_files/wer_model_x.txt", 
        file_b="test_files/wer_model_y.txt", #better model
        total_batch=10000,
        sep="|",
    )
    
    absolute_wer_diff, change_wer_bootstrap, ci_wer  = si_obj.compute_significance_wer(num_samples_per_batch=40, 
                                                                                       ci=confidence_interval)
    
    print(f"StatisticalSignificance: low={change_wer_bootstrap-ci_wer}, high={change_wer_bootstrap+ci_wer}, std_err={ci_wer/1.96}")
    
    # compare to scipy bootstrap
    data = si_obj.data_wer
    res = bootstrap((data,), func, confidence_level=confidence_interval, 
                    method="bca", vectorized=True, batch=10000)
    print(f"Scipy: low={res.confidence_interval.low}, high={res.confidence_interval.high}, std_err={res.standard_error}")
    
    assert np.sign(change_wer_bootstrap-ci_wer) == np.sign(res.confidence_interval.low)
    assert np.sign(change_wer_bootstrap+ci_wer) == np.sign(res.confidence_interval.high)
    
    print("Test passed !!!")
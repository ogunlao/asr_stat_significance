# ASR Statistical Significance

Perform statistical testing on ASR models using the bootstrap method. This library computes the confidence interval (CI) between the WER difference of two competing Automatic speech recognition (ASR) models X and Y.

The implementation is based on these two papers; [Statistical Testing on ASR Performance via Blockwise Bootstrap](https://ieeexplore.ieee.org/abstract/document/1326009) and [Statistical Testing on ASR Performance via Blockwise Bootstrap](https://arxiv.org/abs/1912.09508).

## How to use

For each model X and Y, compute total errors and count of words for each sentence in your test set. Then save them in a text file with a seperator. For example, if sentence 1 has total errors of 5 while the total number of words is 20, it is saved with 5|20 where "|" is used as a separator. You can checkout the `test_files` in this repo for an example.

To find out if model Y is better than model X;

```python
    import numpy as np
    np.random.seed(42)
    
    from asr_stat_significance import StatisticalSignificance

    si_obj = StatisticalSignificance(
        file_a="wer_file_for_model_x.txt", 
        file_b="wer_file_for_model_y.txt", # better model
        total_batch=1000,
    )


    absolute_wer_diff, change_wer_bootstrap, ci_wer  = si_obj.compute_significance_wer(
                        num_samples_per_batch=10000, ci=0.95)
    
    print(f"For WER: {absolute_wer_diff}, low={change_wer_bootstrap-ci_wer},
            high={change_wer_bootstrap+ci_wer}, std={ci_wer/1.96}")

```

If the confidence intervals' (CI) low and high values computed lie fully on the negative side on the real axis (i.e., if both values are negative), then the difference in WER is statistically significant, therefore, model Y is better than model X.

## Function parameters

- ci: Confidence Interval to be used for computation. Typical CI include 90%, 95% and 99%. Default is 95%.
- total_batch: total amount of bootstrap sampling runs. Note that sampling is done with replacement. Typical values are 10^2, 10^3, 10^4. Default is 10^3.
- num_samples_per_batch: The number of WER/CER samples selected from the files per model. This is based on the size of the test set.

## TODO

- Allow bootstrap sampling to be performed based on some criteria such as speakers, gender, or age.

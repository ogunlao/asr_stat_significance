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
        file_path="wer_file.txt", 
        total_batch=1000,
        use_gaussian_appr=True,
    )
    ci_obj  = si_obj.compute_significance(
                        num_samples_per_batch=30, ci=0.95)
    print(ci_obj)
    print(f"The difference in WER between Model X and Y is significant: ", {ci_obj.is_significant()})

    # bootstrap sampling to be performed based on some criteria such as speakers, gender, or age.
    si_obj_block = StatisticalSignificance(
        file_path="wer_file_block.txt", 
        total_batch=1000,
        use_gaussian_appr=True,
    )
    ci_obj_block  = si_obj_block.compute_significance(num_samples_per_batch=30, 
                                                ci=0.95, use_blockwise_bootstrap=True,)
    print(ci_obj_block)
    print(f"The difference in WER between Model X and Y is significant: ", {ci_obj_block.is_significant()})

```

## How to interprete the CI values

If the confidence intervals' (CI) low and high values computed lie fully on the negative side on the real axis (i.e., if both values are negative), then the difference in WER is statistically significant. Then, one can say model Y is better than model X.

## Function parameters

- ci: Confidence Interval to be used for computation. Typical CI include 90%, 95% and 99%. Default is 95%.
- total_batch: total amount of bootstrap sampling runs. Note that sampling is done with replacement. Typical values are 10^2, 10^3, 10^4. Default is 10^3.
- num_samples_per_batch: The number of WER/CER data selected to compute the mean. If `use_blockwise_bootstrap=True`, then, we compute `num_samples_per_block=num_samples_per_batch//total_num_of_blocks`. Value should be based on the size of the test set.

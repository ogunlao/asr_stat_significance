# WER Statistical Significance Testing

Perform statistical testing on word error rates between ASR models using the bootstrap method. This library computes the confidence interval (CI) between the WER difference of two competing Automatic speech recognition (ASR) models X and Y.

The implementation is based on these two papers; [Statistical Testing on ASR Performance via Blockwise Bootstrap](https://ieeexplore.ieee.org/abstract/document/1326009) and [Statistical Testing on ASR Performance via Blockwise Bootstrap](https://arxiv.org/abs/1912.09508).

## How to use

For each model X and Y, compute total errors and count of words for each sentence in your test set. Then save them in a text file with a seperator. For example, for a sentence in the test set;

model A: 5 errors
model B: 10 errors
total number of words in sentence: 12

Save as `5|10|12` where "|" is used as a separator. If you intend to use blockwise sampling, e.g. sampling on speakers, then save as `5|10|12|speaker_id`

Checkout the `test_files` directory for example files.

To find out if model Y is better than model X;

```python
    import numpy as np
    np.random.seed(42)
    
    cd wer_stat_significance
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
```

OR bootstrap sampling to be performed based on some criteria such as speakers, gender, or age.

```python
    import numpy as np
    np.random.seed(42)
    
    from asr_stat_significance import StatisticalSignificance

    si_obj_block = StatisticalSignificance(
        file_path="wer_file_block.txt", 
        total_batch=1000,
        use_gaussian_appr=True,
    )
    ci_obj_block  = si_obj_block.compute_significance(num_samples_per_block=2, 
                                                ci=0.95, use_blockwise_bootstrap=True,)
    print(ci_obj_block)
    print(f"The difference in WER between Model X and Y is significant: ", {ci_obj_block.is_significant()})

```

## How to interprete the CI values

If the computed CI low and CI high values lie fully on the negative side on the real axis (i.e., if both values are negative), then, the difference in WER is statistically significant. Therefore, one can say model Y is better than model X.

## Function parameters

- ci: Confidence Interval to be used for computation. Typical CI include 90%, 95% and 99%. Default is 95%.
- total_batch: total amount of bootstrap sampling runs. Note that sampling is done with replacement. Typical values are 10^2, 10^3, 10^4. Default is 10^3.
- num_samples_per_batch: The number of WER/CER data selected for each batch. Value should be based on the size of the test set.
- num_samples_per_block: If `use_blockwise_bootstrap=True`, then, this is the number of samples to collect from each block during bootstrap sampling.

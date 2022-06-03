# DebiasedBERT4Rec
Pytorch implementation of the paper "Debiasing the Cloze Task in Sequential Recommendation with Bidirectional Transformers".

Accepted at KDD '22.

Link to paper: 

## Authors
Khalil Damak, University of Louisville.<br>
Sami Khenissi, University of Louisville.<br>
Olfa Nasraoui, University of Louisville.<br>

## Abstract
Bidirectional Transformer architectures are state-of-the-art sequential recommendation models that use a bi-directional representation capacity based on the Cloze task, a.k.a. Masked Language Model. The latter aims to predict randomly masked items within the sequence. Because they assume that the true interacted item is the most relevant one, an exposure bias results, where non-interacted items with low exposure propensities are assumed to be irrelevant. The most common approach to mitigating exposure bias in recommendation has been Inverse Propensity Scoring (IPS), which consists of down-weighting the interacted predictions in the loss function in proportion to their propensities of exposure, yielding a theoretically unbiased learning.
In this work, we argue and prove that IPS does not extend to sequential recommendation because it fails to account for the temporal nature of the problem. We then propose a novel propensity scoring mechanism, which can theoretically debias the Cloze task in sequential recommendation. Finally we empirically demonstrate the debiasing capabilities of our proposed approach and its robustness to the severity of exposure bias.

## Environment settings
We use Pytorch 1.7.1.

## Citation


## Description
To run the code, use the following command:

```
python main.py [-h] 
               [--mode {train_bert_real,tune_bert_real,loop_bert_real,train_bert_semi_synthetic,tune_bert_semi_synthetic,generate_semi_synthetic,train_tf,tune_tf}]
               [--template TEMPLATE] 
               [--dataset_code {ml-1m,ml-20m,ml-100k,amazon-beauty}] 
               [--min_rating MIN_RATING] 
               [--min_uc MIN_UC] 
               [--min_sc MIN_SC] 
               [--split {leave_one_out}] 
               [--dataset_split_seed DATASET_SPLIT_SEED] 
               [--generate_semi_synthetic_seed GENERATE_SEMI_SYNTHETIC_SEED] 
               [--dataloader_random_seed DATALOADER_RANDOM_SEED]
               [--train_batch_size TRAIN_BATCH_SIZE] 
               [--val_batch_size VAL_BATCH_SIZE] 
               [--test_batch_size TEST_BATCH_SIZE] 
               [--train_negative_sampler_code {popular,random}]
               [--train_negative_sample_size TRAIN_NEGATIVE_SAMPLE_SIZE] 
               [--train_negative_sampling_seed TRAIN_NEGATIVE_SAMPLING_SEED] 
               [--test_negative_sampler_code {popular,random}]
               [--test_negative_sample_size TEST_NEGATIVE_SAMPLE_SIZE] 
               [--test_negative_sampling_seed TEST_NEGATIVE_SAMPLING_SEED] 
               [--device {cpu,cuda}] 
               [--num_gpu NUM_GPU]
               [--device_idx DEVICE_IDX] 
               [--optimizer {SGD,Adam}] 
               [--lr LR] [--weight_decay WEIGHT_DECAY] 
               [--momentum MOMENTUM] 
               [--enable_lr_schedule ENABLE_LR_SCHEDULE]
               [--decay_step DECAY_STEP] 
               [--gamma GAMMA] 
               [--num_iterations NUM_ITERATIONS] 
               [--num_epochs NUM_EPOCHS] 
               [--log_period_as_iter LOG_PERIOD_AS_ITER]
               [--metric_ks METRIC_KS [METRIC_KS ...]] 
               [--best_metric BEST_METRIC] 
               [--top_k_recom TOP_K_RECOM] 
               [--model_init_seed MODEL_INIT_SEED] 
               [--bert_max_len BERT_MAX_LEN]
               [--bert_hidden_units BERT_HIDDEN_UNITS] 
               [--bert_num_blocks BERT_NUM_BLOCKS] 
               [--bert_num_heads BERT_NUM_HEADS] 
               [--bert_dropout BERT_DROPOUT]
               [--bert_mask_prob BERT_MASK_PROB] 
               [--skew_power SKEW_POWER]
               [--loss_debiasing {None,static_propensity,temporal_propensity,relevance,static_popularity,temporal_popularity}] 
               [--propensity_clipping PROPENSITY_CLIPPING]
               [--tf_hidden_units TF_HIDDEN_UNITS] 
               [--tf_target {exposure,relevance}] 
               [--frac_exposure_negatives FRAC_EXPOSURE_NEGATIVES] 
               [--skewness_parameter SKEWNESS_PARAMETER]
               [--unbiased_eval UNBIASED_EVAL] 
               [--experiment_dir EXPERIMENT_DIR] 
               [--experiment_description EXPERIMENT_DESCRIPTION] 
               [--num_configurations NUM_CONFIGURATIONS]
               [--num_reps NUM_REPS] 
               [--tune_bert_hidden_units TUNE_BERT_HIDDEN_UNITS] 
               [--tune_bert_num_blocks TUNE_BERT_NUM_BLOCKS] 
               [--tune_bert_num_heads TUNE_BERT_NUM_HEADS]
               [--tune_train_batch_size TUNE_TRAIN_BATCH_SIZE] 
               [--tune_bert_dropout TUNE_BERT_DROPOUT] 
               [--tune_bert_mask_prob TUNE_BERT_MASK_PROB] 
               [--tune_skew_power TUNE_SKEW_POWER]
               [--tune_tf_hidden_units TUNE_TF_HIDDEN_UNITS]
```

You can perform the following tasks with the argument "mode":
* Train BERT4Rec [1] on a real-world dataset: use --mode train_bert_real
* Tune BERT4Rec on a real-world dataset: use --mode tune_bert_real
* Train BERT4Rec on a feedback loop, on a real-world dataset: use --mode loop_bert_real
* Train BERT4Rec on a semi-synthetic dataset: use --mode train_bert_semi_synthetic
* Tune BERT4Rec on a semi-synthetic dataset: use --mode tune_bert_semi_synthetic
* Train a Tensor Factorization model on a real-world dataset: use --train_tf
* Tune a Tensor Factorization model on a real-world dataset: use --tune_tf
* Generate semi-synthetic data: use --mode generate_semi_synthetic with --tf_target exposure to generate interactions, and --tf_target relevance to generate ratings.

To use a debiased loss function, use the argument --loss_debiasing with the values:
* 'static_propensity' for IPS-BERT4Rec with semi-synthetic propensities.
* 'temporal_propensity' for ITPS-BERT4Rec with semi-synthetic propensities.
* 'relevance' for the Oracle with semi-synthetic relevance.
* 'static_popularity' for IPS-BERT4Rec with real popularities.
* 'temporal_popularity' is ITPS-BERT4Rec with real temporal popularities.

All argument descriptions are below:

```
optional arguments:
  -h, --help            show this help message and exit
  --mode {train_bert_real,tune_bert_real,loop_bert_real,train_bert_semi_synthetic,tune_bert_semi_synthetic,generate_semi_synthetic,train_tf,tune_tf}
  --template TEMPLATE
  --dataset_code {ml-1m,ml-20m,ml-100k,amazon-beauty}
  --min_rating MIN_RATING
                        Only keep ratings greater than or equal to this value.
  --min_uc MIN_UC       Only keep users with more than min_uc ratings.
  --min_sc MIN_SC       Only keep items with more than min_sc ratings.
  --split {leave_one_out}
                        How to split the datasets.
  --dataset_split_seed DATASET_SPLIT_SEED
  --generate_semi_synthetic_seed GENERATE_SEMI_SYNTHETIC_SEED
  --dataloader_random_seed DATALOADER_RANDOM_SEED
  --train_batch_size TRAIN_BATCH_SIZE
  --val_batch_size VAL_BATCH_SIZE
  --test_batch_size TEST_BATCH_SIZE
  --train_negative_sampler_code {popular,random}
                        Method to sample negative items for training. Not needed in BERT
  --train_negative_sample_size TRAIN_NEGATIVE_SAMPLE_SIZE
  --train_negative_sampling_seed TRAIN_NEGATIVE_SAMPLING_SEED
  --test_negative_sampler_code {popular,random}
                        Method to sample negative items for evaluation
  --test_negative_sample_size TEST_NEGATIVE_SAMPLE_SIZE
  --test_negative_sampling_seed TEST_NEGATIVE_SAMPLING_SEED
  --device {cpu,cuda}
  --num_gpu NUM_GPU
  --device_idx DEVICE_IDX
  --optimizer {SGD,Adam}
  --lr LR               Learning rate.
  --weight_decay WEIGHT_DECAY
                        l2 regularization.
  --momentum MOMENTUM   SGD momentum.
  --enable_lr_schedule ENABLE_LR_SCHEDULE
                        Set True to enable learning rate decay.
  --decay_step DECAY_STEP
                        Decay step for StepLR.
  --gamma GAMMA         Gamma for StepLR.
  --num_iterations NUM_ITERATIONS
                        Number of feedback loop iterations.
  --num_epochs NUM_EPOCHS
                        Number of epochs for training.
  --log_period_as_iter LOG_PERIOD_AS_ITER
  --metric_ks METRIC_KS [METRIC_KS ...]
                        List of cutoffs k in Metrics@k.
  --best_metric BEST_METRIC
                        Metric used for optimizing the model.
  --top_k_recom TOP_K_RECOM
                        Number of recommended items at each iteration from which the user is assumed to randomly select one. Used in the feedback loop.
  --model_init_seed MODEL_INIT_SEED
  --bert_max_len BERT_MAX_LEN
                        Max sequence length.
  --bert_hidden_units BERT_HIDDEN_UNITS
                        Size of hidden vectors (d_model) in BERT.
  --bert_num_blocks BERT_NUM_BLOCKS
                        Number of transformer layers.
  --bert_num_heads BERT_NUM_HEADS
                        Number of heads for multi-head attention.
  --bert_dropout BERT_DROPOUT
                        Dropout probability to use throughout the model.
  --bert_mask_prob BERT_MASK_PROB
                        Probability for masking items in the training sequence.
  --skew_power SKEW_POWER
                        Skewing power applied on propensities to scale them and avoid numerical overflow.
  --loss_debiasing {None,static_propensity,temporal_propensity,relevance,static_popularity,temporal_popularity}
                        Type of debiasing to apply on the loss: None is BERT4Rec, 'static_propensity' is IPS-BERT4Rec with semi-synthetic propensities, 'temporal_propensity' is ITPS-
                        BERT4Rec with semi-synthetic propensities, 'relevance' is Oracle-BERT4Rec with semi-synthetic relevance, 'static_popularity' is IPS-BERT4Rec with real
                        popularities, 'temporal_popularity' is ITPS-BERT4Rec with real temporal popularities.
  --propensity_clipping PROPENSITY_CLIPPING
                        Propensity clipping value used as lower threshold to avoid numerical precision issues when propensity values are too small.
  --tf_hidden_units TF_HIDDEN_UNITS
                        Number of hidden units in the Tensor Factorization model.
  --tf_target {exposure,relevance}
                        Target to be modeled by the Tensor Factorization model.
  --frac_exposure_negatives FRAC_EXPOSURE_NEGATIVES
                        Fraction of sampled instances with negative exposure per number of positive instances. If None, all non-interactions will be considered as having negative
                        exposure.
  --skewness_parameter SKEWNESS_PARAMETER
                        Power applied to the propensity scores to control the exposure bias through the skewness of the distribution. (Power p in the paper.)
  --unbiased_eval UNBIASED_EVAL
                        Set to True to enable: 1) an unbiased semi-synthetic evaluation on true relevance if mode in ["train_bert_semi_synthetic", "tune_bert_semi_synthetic"]; or 2) an
                        unbiased evaluation on real data using ITPS-based evaluation metrics if mode in ["train_bert_real", "tune_bert_real", "loop_bert_real"].
  --experiment_dir EXPERIMENT_DIR
  --experiment_description EXPERIMENT_DESCRIPTION
  --num_configurations NUM_CONFIGURATIONS
                        Number of random hyperparameter configurations.
  --num_reps NUM_REPS   Number of replicates in hyperparameter tuning.
  --tune_bert_hidden_units TUNE_BERT_HIDDEN_UNITS
                        Tuning values for bert_hidden_units.
  --tune_bert_num_blocks TUNE_BERT_NUM_BLOCKS
                        Tuning values for bert_num_blocks.
  --tune_bert_num_heads TUNE_BERT_NUM_HEADS
                        Tuning values for bert_num_heads.
  --tune_train_batch_size TUNE_TRAIN_BATCH_SIZE
                        Tuning values for train_batch_size.
  --tune_bert_dropout TUNE_BERT_DROPOUT
                        Tuning values for bert_dropout.
  --tune_bert_mask_prob TUNE_BERT_MASK_PROB
                        Tuning values for bert_mask_prob.
  --tune_skew_power TUNE_SKEW_POWER
                        Tuning values for skew_power.
  --tune_tf_hidden_units TUNE_TF_HIDDEN_UNITS
                        Tuning values for tf_hidden_units.
```

## Templates
We provide the following templates that can be used to reproduce the experiments in the paper:

* Train the different models on semi-synthetic data and evaluate them using the unbiased evaluation process **(RQ1)**:
```
python -m main --template train_bert_semi_synthetic_unbiased_eval
python -m main --template train_ips_bert_semi_synthetic_unbiased_eval
python -m main --template train_itps_bert_semi_synthetic_unbiased_eval
python -m main --template train_oracle_semi_synthetic_unbiased_eval
```

* Train the different models on semi-synthetic data with a skewed exposure distribution, and evaluate them with the unbiased evaluation process. The exposure skewness power is input by the user **(RQ2)**:
```
python -m main --template robustness_increasing_bias_bert_semi_synthetic_unbiased_eval
python -m main --template robustness_increasing_bias_ips_bert_semi_synthetic_unbiased_eval
python -m main --template robustness_increasing_bias_itps_bert_semi_synthetic_unbiased_eval
python -m main --template robustness_increasing_bias_oracle_semi_synthetic_unbiased_eval
```

* Train the different models on semi-synthetic data and evaluate them with the standard LOO evaluation process **(RQ3)**:
```
python -m main --template train_bert_semi_synthetic_biased_eval
python -m main --template train_ips_bert_semi_synthetic_biased_eval
python -m main --template train_itps_bert_semi_synthetic_biased_eval
python -m main --template train_oracle_semi_synthetic_biased_eval
```

* Tune the different models on semi-synthetic data. The hyperparameter tuning is performed on a validation set using the unbiased evaluation process:
```
python -m main --template tune_bert_semi_synthetic_unbiased_eval
python -m main --template tune_ips_bert_semi_synthetic_unbiased_eval
python -m main --template tune_itps_bert_semi_synthetic_unbiased_eval
python -m main --template tune_oracle_semi_synthetic_unbiased_eval
```

* Generate semi-synthetic data using Tensor Factorization:
```
python -m main --template generate_semi_synthetic_data_interactions
python -m main --template generate_semi_synthetic_data_ratings
```

* Tune the Tensor Factorization model for the tasks of interaction and rating prediction respectively. The hyperparameter tuning is performed on a validation set.
```
python -m main --template tune_tf_semi_synthetic_interactions
python -m main --template tune_tf_semi_synthetic_ratings
```

* Train the different models on the different datasets and evaluate them on test sets where the negative samples are sampled based on their popularity to mitigate exposure bias in the evaluation **(RQ4)**:
```
python -m main --template train_bert_real_ml_1m_pop_eval_sampling
python -m main --template train_ips_bert_real_ml_1m_pop_eval_sampling
python -m main --template train_itps_bert_real_ml_1m_pop_eval_sampling
python -m main --template train_bert_real_ml_20m_pop_eval_sampling
python -m main --template train_ips_bert_real_ml_20m_pop_eval_sampling
python -m main --template train_itps_bert_real_ml_20m_pop_eval_sampling
python -m main --template train_bert_real_amazon_beauty_pop_eval_sampling
python -m main --template train_ips_bert_real_amazon_beauty_pop_eval_sampling
python -m main --template train_itps_bert_real_amazon_beauty_pop_eval_sampling
```

* Tune the different models on the different datasets. The tuning is performed on a validation set in which the negative interactions are sampled based on popularity to mitigate exposure bias in the evaluation:
```
python -m main --template tune_bert_real_ml_1m_pop_eval_sampling
python -m main --template tune_ips_bert_real_ml_1m_pop_eval_sampling
python -m main --template tune_itps_bert_real_ml_1m_pop_eval_sampling
python -m main --template tune_bert_real_ml_20m_pop_eval_sampling
python -m main --template tune_ips_bert_real_ml_20m_pop_eval_sampling
python -m main --template tune_itps_bert_real_ml_20m_pop_eval_sampling
python -m main --template tune_bert_real_amazon_beauty_pop_eval_sampling
python -m main --template tune_ips_bert_real_amazon_beauty_pop_eval_sampling
python -m main --template tune_itps_bert_real_amazon_beauty_pop_eval_sampling
```

* Simulate a feedback loop with recommendations from the different models on the different datasets. The ranking evaluation is based on popularity negative sampling **(RQ5)**:
```
python -m main --template loop_bert_real_ml_1m_pop_eval_sampling
python -m main --template loop_ips_bert_real_ml_1m_pop_eval_sampling
python -m main --template loop_itps_bert_real_ml_1m_pop_eval_sampling
python -m main --template loop_bert_real_ml_20m_pop_eval_sampling
python -m main --template loop_ips_bert_real_ml_20m_pop_eval_sampling
python -m main --template loop_itps_bert_real_ml_20m_pop_eval_sampling
python -m main --template loop_bert_real_amazon_beauty_pop_eval_sampling
python -m main --template loop_ips_bert_real_amazon_beauty_pop_eval_sampling
python -m main --template loop_itps_bert_real_amazon_beauty_pop_eval_sampling
```

## Datasets
We provide code ready to run on the:

* Movielens 100K dataset.
* Movielens 1M dataset.
* Movielens 20M dataset. (due to a file size limit, you need to [download](https://grouplens.org/datasets/movielens/20m/) the dataset and add it to a folder "Data/ml-20m" to be able to use it.)
* Amazon Beauty dataset.

## References

[1] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM '19). Association for Computing Machinery, New York, NY, USA, 1441–1450. https://doi.org/10.1145/3357384.3357895

## Acknowledgements
This repo https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch was a good starting point for our project. We would like to thank the contributors of the repo for making it available.

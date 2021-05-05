# DebiasedBERT4Rec
Pytorch implementation of the paper "Debiasing the Cloze Task in Session-Based Recommendation with Bidirectional Transformers".

This is an anonymous copy of the private Github repository for RecSys submission.

## Environment settings
We use Pytorch 1.7.1.

## Description
To run the code, use the following command:

```
python -m main [-h]
               [--mode {train_bert_real,tune_bert_real,loop_bert_real,train_bert_semi_synthetic,tune_bert_semi_synthetic,generate_semi_synthetic,train_tf,tune_tf}]
               [--template TEMPLATE] [--test_model_path TEST_MODEL_PATH]
               [--dataset_code {ml-1m,ml-20m,ml-100k,amazon-beauty}]
               [--min_rating MIN_RATING] [--min_uc MIN_UC] [--min_sc MIN_SC]
               [--split SPLIT] [--dataset_split_seed DATASET_SPLIT_SEED]
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
               [--device {cpu,cuda}] [--num_gpu NUM_GPU]
               [--device_idx DEVICE_IDX] [--optimizer {SGD,Adam}] [--lr LR]
               [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
               [--enable_lr_schedule ENABLE_LR_SCHEDULE]
               [--decay_step DECAY_STEP] [--gamma GAMMA]
               [--num_iterations NUM_ITERATIONS] [--num_epochs NUM_EPOCHS]
               [--log_period_as_iter LOG_PERIOD_AS_ITER]
               [--metric_ks METRIC_KS [METRIC_KS ...]]
               [--best_metric BEST_METRIC] [--top_k_recom TOP_K_RECOM]
               [--model_init_seed MODEL_INIT_SEED]
               [--bert_max_len BERT_MAX_LEN] [--bert_num_items BERT_NUM_ITEMS]
               [--bert_hidden_units BERT_HIDDEN_UNITS]
               [--bert_num_blocks BERT_NUM_BLOCKS]
               [--bert_num_heads BERT_NUM_HEADS] [--bert_dropout BERT_DROPOUT]
               [--bert_mask_prob BERT_MASK_PROB] [--skew_power SKEW_POWER]
               [--loss_debiasing {None,static_propensity,temporal_propensity,relevance,static_popularity,temporal_popularity}]
               [--tf_hidden_units TF_HIDDEN_UNITS]
               [--tf_target {exposure,relevance}]
               [--frac_exposure_negatives FRAC_EXPOSURE_NEGATIVES]
               [--skewness_parameter SKEWNESS_PARAMETER]
               [--unbiased_eval UNBIASED_EVAL]
               [--experiment_dir EXPERIMENT_DIR]
               [--experiment_description EXPERIMENT_DESCRIPTION]
               [--num_configurations NUM_CONFIGURATIONS] [--num_reps NUM_REPS]
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
* Train BERT4Rec on a real-world dataset: use --mode train_bert_real
* Tune BERT4Rec on a real-world dataset: use --mode tune_bert_real
* Train BERT4Rec on a feedback loop, on a real-world dataset: use --mode loop_bert_real
* Train BERT4Rec on a semi-synthetic dataset: use --mode train_bert_semi_synthetic
* Tune BERT4Rec on a semi-synthetic dataset: use --mode tune_bert_semi_synthetic
* Train Tensor Factorization model on a real-world dataset: use --train_tf
* Tune Tensor Factorization model on a real-world dataset: use --tune_tf
* Generate semi-synthetic data: use --mode generate_semi_synthetic with --tf_target exposure to generate interactions, and --tf_target relevance to generate ratings.

To use a debiased loss function, use the argument --loss_debiasing with the values:
* 'static_propensity' for IPS-BERT4Rec with semi-synthetic propensities.
* 'temporal_propensity' for ITPS-BERT4Rec with semi-synthetic propensities.
* 'relevance' for Rel-BERT4Rec with semi-synthetic relevance.
* 'static_popularity' for IPS-BERT4Rec with real popularities.
* 'temporal_popularity' is ITPS-BERT4Rec with real temporal popularities.

All argument descriptions are below:

```
optional arguments:
  -h, --help            show this help message and exit
  --mode {train_bert_real,tune_bert_real,loop_bert_real,train_bert_semi_synthetic,tune_bert_semi_synthetic,generate_semi_synthetic,train_tf,tune_tf}
  --template TEMPLATE
  --test_model_path TEST_MODEL_PATH
  --dataset_code {ml-1m,ml-20m,ml-100k,amazon-beauty}
  --min_rating MIN_RATING
                        Only keep ratings greater than equal to this value
  --min_uc MIN_UC       Only keep users with more than min_uc ratings
  --min_sc MIN_SC       Only keep items with more than min_sc ratings
  --split SPLIT         How to split the datasets
  --dataset_split_seed DATASET_SPLIT_SEED
  --generate_semi_synthetic_seed GENERATE_SEMI_SYNTHETIC_SEED
  --dataloader_random_seed DATALOADER_RANDOM_SEED
  --train_batch_size TRAIN_BATCH_SIZE
  --val_batch_size VAL_BATCH_SIZE
  --test_batch_size TEST_BATCH_SIZE
  --train_negative_sampler_code {popular,random}
                        Method to sample negative items for training. Not
                        needed in BERT
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
  --lr LR               Learning rate
  --weight_decay WEIGHT_DECAY
                        l2 regularization
  --momentum MOMENTUM   SGD momentum
  --enable_lr_schedule ENABLE_LR_SCHEDULE
                        Set True to enable learning rate decay.
  --decay_step DECAY_STEP
                        Decay step for StepLR
  --gamma GAMMA         Gamma for StepLR
  --num_iterations NUM_ITERATIONS
                        Number of feedback loop iterations.
  --num_epochs NUM_EPOCHS
                        Number of epochs for training.
  --log_period_as_iter LOG_PERIOD_AS_ITER
  --metric_ks METRIC_KS [METRIC_KS ...]
                        ks for Metric@k
  --best_metric BEST_METRIC
                        Metric for determining the best model.
  --top_k_recom TOP_K_RECOM
                        Number of recommended items at each iteration from
                        which the user is assumed to randomly select one.
  --model_init_seed MODEL_INIT_SEED
  --bert_max_len BERT_MAX_LEN
                        Max sequence length.
  --bert_num_items BERT_NUM_ITEMS
                        Number of total items.
  --bert_hidden_units BERT_HIDDEN_UNITS
                        Size of hidden vectors (d_model) in BERT.
  --bert_num_blocks BERT_NUM_BLOCKS
                        Number of transformer layers.
  --bert_num_heads BERT_NUM_HEADS
                        Number of heads for multi-head attention.
  --bert_dropout BERT_DROPOUT
                        Dropout probability to use throughout the model.
  --bert_mask_prob BERT_MASK_PROB
                        Probability for masking items in the training
                        sequence.
  --skew_power SKEW_POWER
                        Skewing power applied on propensities to scale them
                        and avoid numerical overflow.
  --loss_debiasing {None,static_propensity,temporal_propensity,relevance,static_popularity,temporal_popularity}
                        Type of debiasing to apply on the loss: None is
                        BERT4Rec, 'static_propensity' is IPS-BERT4Rec with
                        semi-synthetic propensities, 'temporal_propensity' is
                        ITPS-BERT4Rec with semi-synthetic propensities,
                        'relevance' is Oracle-BERT4Rec with semi-synthetic
                        relevance, 'static_popularity' is IPS-BERT4Rec with
                        real popularities, 'temporal_popularity' is ITPS-
                        BERT4Rec with real temporal popularities.
  --tf_hidden_units TF_HIDDEN_UNITS
                        Number of hidden units in the Tensor Factorization
                        model.
  --tf_target {exposure,relevance}
                        Target to be modeled by the Tensor Factorization
                        model.
  --frac_exposure_negatives FRAC_EXPOSURE_NEGATIVES
                        Fraction of sampled instances with negative exposure
                        per number of positive instances. If None, all non-
                        interactions will be considered as having negative
                        exposure.
  --skewness_parameter SKEWNESS_PARAMETER
                        Power applied to the propensity scores to control the
                        exposure bias through the skewness of the
                        distribution.
  --unbiased_eval UNBIASED_EVAL
                        Set True to enable an unbiased semi-synthetic
                        evaluation on true relevance.
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

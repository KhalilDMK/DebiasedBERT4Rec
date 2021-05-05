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
* Train BERT4Rec on a real-world dataset: use <script>--mode train_bert_real</script>

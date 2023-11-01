#!/bin/sh
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name="strategy_classifier"
#SBATCH --output=log4.out
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=rohini
#SBATCH --gres=gpu:2

echo "submitting job..."
WANDB_ENTITY=navidmdn WANDB_PROJECT=strategy_classifier PYTHONPATH=. python train_strategy_classifier.py --cache_dir "../hfcache" --model_name_or_path\
 roberta-base --train_file original_data/sfe_train.json --validation_file original_data/sfe_valid.json --shuffle_train_dataset --metric_name f1\
 --text_column_name context --label_column_name strategy --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 16\
 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 --output_dir output/roberta_strategy_classifier\
 --overwrite_output_dir --evaluation_strategy steps --logging_strategy steps --logging_steps 25 --eval_steps 100\
 --metric_for_best_model loss --load_best_model_at_end --save_total_limit 1 --save_steps 300
echo "job finished successfully."


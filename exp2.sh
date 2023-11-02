#!/bin/sh
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name="strategy_follower"
#SBATCH --output=exp2.out
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=rohini
#SBATCH --gres=gpu:2

echo "submitting job..."
WANDB_ENTITY=navidmdn WANDB_PROJECT=strategy_follower PYTHONPATH=. python strategy_conditioned_utterance_generation_trainer.py --cache_dir "../hfcache" \
--train_file original_data/train_extes_strategy.json --strategy_file original_data/extes_strategy.json \
--validation_file original_data/valid_extes_strategy.json --do_train --model_name_or_path facebook/bart-large --output_dir ./output/exp2 --overwrite_output_dir \
--preprocessing_num_workers 4 --num_train_epochs 10 --evaluation_strategy steps --logging_strategy steps --logging_steps 50 --predict_with_generate \
--eval_steps 300 --per_device_train_batch_size 16 --metric_for_best_model loss --load_best_model_at_end --save_total_limit 1 --save_steps 300
echo "job finished successfully."

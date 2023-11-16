#!/bin/sh
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name="joint-strategy-utt"
#SBATCH --output=log1.out
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=rohini
#SBATCH --gres=gpu:2

echo "submitting job..."
WANDB_ENTITY=navidmdn WANDB_PROJECT=joint_utt_strategy PYTHONPATH=. python joint_strategy_response_trainer.py --cache_dir "../hfcache" --train_file data/preprocessed_train.json \
--validation_file data/preprocessed_valid.json --do_train --model_name_or_path facebook/bart-base --output_dir ./output/bart-base --overwrite_output_dir \
--preprocessing_num_workers 4 --num_train_epochs 10 --evaluation_strategy steps --logging_strategy steps --logging_steps 50 --predict_with_generate \
--eval_steps 300 --per_device_train_batch_size 16
echo "job finished successfully."

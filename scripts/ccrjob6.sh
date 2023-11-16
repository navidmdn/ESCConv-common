#!/bin/sh
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name="ESC"
#SBATCH --output=log5.out
#SBATCH --cluster=faculty
#SBATCH --partition=kjoseph
#SBATCH --qos=kjoseph
#SBATCH --account=kjoseph
#SBATCH --gres=gpu:2

echo "submitting job..."
WANDB_ENTITY=navidmdn WANDB_PROJECT=peft_strategy_follower PYTHONPATH=. python peft_prompt_tuning_clm.py \
--cache_dir "../hfcache" \
--train_file original_data/merged.json --strategy_file original_data/extes_strategy.json \
--validation_file original_data/valid.json --model_name gpt2-large --output_dir \
./output/gpt2large-prompt-tuning --load_in_8bit
echo "job finished successfully."


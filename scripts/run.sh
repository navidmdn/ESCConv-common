WANDB_ENTITY=navidmdn WANDB_PROJECT=strategy_follower PYTHONPATH=. python peft_prompt_tuning_clm.py \
--cache_dir "../hfcache" \
--train_file original_data/merged.json --strategy_file original_data/extes_strategy.json \
--validation_file original_data/valid.json --do_train --model_name_or_path ../../models/llama2-7b-hf --output_dir \
./output/llamav2-prompt-tuning-exp4 --overwrite_output_dir --load_in_8bit

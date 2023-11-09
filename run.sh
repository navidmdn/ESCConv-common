WANDB_ENTITY=navidmdn WANDB_PROJECT=strategy_follower PYTHONPATH=. python peft_prompt_tuning_clm.py \
--cache_dir "../hfcache" \
--train_file original_data/train.json --strategy_file original_data/esconv_strategy.json \
--validation_file original_data/valid.json --do_train --model_name_or_path ../../models/llama2-7b-hf --output_dir ./output/llama --overwrite_output_dir \
--preprocessing_num_workers 4 --num_train_epochs 10 --evaluation_strategy steps --logging_strategy steps --logging_steps 10 \
--eval_steps 50 --per_device_train_batch_size 1 --metric_for_best_model loss --load_best_model_at_end --save_total_limit 1 --save_steps 300 --fp16 \
--max_source_length 256 --gradient_accumulation_steps 4

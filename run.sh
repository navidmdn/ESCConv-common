WANDB_ENTITY=navidmdn WANDB_PROJECT=strategy_follower PYTHONPATH=. python strategy_conditioned_utterance_generation_trainer.py \
--train_file data/ExTES/train.json --strategy_file original_data/extes_strategy.json \
--validation_file data/ExTES/valid.json --do_train --model_name_or_path facebook/bart-base --output_dir ./output/exp3 --overwrite_output_dir \
--preprocessing_num_workers 4 --num_train_epochs 10 --evaluation_strategy steps --logging_strategy steps --logging_steps 50 --predict_with_generate \
--eval_steps 300 --per_device_train_batch_size 16 --metric_for_best_model loss --load_best_model_at_end --save_total_limit 1 --save_steps 300
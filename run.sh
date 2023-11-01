WANDB_ENTITY=navidmdn WANDB_PROJECT=ExTES PYTHONPATH=. python strategy_classifier_trainer.py --cache_dir "../hfcache" --model_name_or_path\
 roberta-base --train_file data/ExTES/sfe_train.json --validation_file data/ExTES/sfe_valid.json --shuffle_train_dataset --metric_name f1\
 --text_column_name context --label_column_name strategy --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 16\
 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 --output_dir output/roberta_strategy_classifier_ExTES\
 --overwrite_output_dir --evaluation_strategy steps --logging_strategy steps --logging_steps 100 --eval_steps 250\
 --metric_for_best_model loss --load_best_model_at_end --save_total_limit 1 --save_steps 500
PYTHONPATH=. WANDB_ENTITY=navidmdn WANDB_PROJECT=llama2chat-convprefix python trainers/peft_llama_with_conversation_prefix.py\
 --train_file original_data/train.json --validation_file original_data/valid.json --output_dir output/llama2chat-convprefix-fo2 \
 --seq_length 400 --model_name ../../models/llama2-7b-chat-hf/ --cache_dir ../hfcache/ --load_in_4bit\
 --log_with wandb --prefix_fanout 2 --num_train_epochs 3

PYTHONPATH=. WANDB_ENTITY=navidmdn WANDB_PROJECT=llama2chat-convprefix python trainers/peft_llama_with_conversation_prefix.py\
 --train_file original_data/train.json --validation_file original_data/valid.json --output_dir output/llama2chat-convprefix-fo4 \
 --seq_length 400 --model_name ../../models/llama2-7b-chat-hf/ --cache_dir ../hfcache/ --load_in_4bit\
 --log_with wandb --prefix_fanout 4 --num_train_epochs 3

PYTHONPATH=. WANDB_ENTITY=navidmdn WANDB_PROJECT=llama2chat-convprefix python trainers/peft_llama_with_conversation_prefix.py\
 --train_file original_data/train.json --validation_file original_data/valid.json --output_dir output/llama2chat-convprefix-fo8 \
 --seq_length 400 --model_name ../../models/llama2-7b-chat-hf/ --cache_dir ../hfcache/ --load_in_4bit\
 --log_with wandb --prefix_fanout 8 --num_train_epochs 3

PYTHONPATH=. WANDB_ENTITY=navidmdn WANDB_PROJECT=llama2chat-convprefix python trainers/peft_llama_with_conversation_prefix.py\
 --train_file original_data/train.json --validation_file original_data/valid.json --output_dir output/llama2chat-convprefix-fo16 \
 --seq_length 400 --model_name ../../models/llama2-7b-chat-hf/ --cache_dir ../hfcache/ --load_in_4bit\
 --log_with wandb --prefix_fanout 16 --num_train_epochs 3




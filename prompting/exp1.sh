PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-7b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp1_7b --get_attentions True --prompt_constructor partial --n_iters 1000

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp1_13b --get_attentions True --prompt_constructor partial --n_iters 500

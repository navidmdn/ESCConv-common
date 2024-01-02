PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-7b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp2_7b --get_attentions True --prompt_constructor full --n_iters 1000

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp2_13b --get_attentions True --prompt_constructor full --n_iters 500

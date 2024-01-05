#PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-7b-chat-hf\
#  --cache_dir ../../hfcache/ --output_path outputs/exp1_7b_responly --get_attentions False --prompt_constructor partial

#PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
#  --cache_dir ../../hfcache/ --output_path outputs/exp1_13b --get_attentions True --prompt_constructor partial --n_iters 500
#
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-70b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp1_70b_responly --get_attentions False --prompt_constructor partial
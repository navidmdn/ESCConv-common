import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict
import fire
from trainers.llama_prompt import modified_extes_support_strategies
from trainers.llama_prompt import B_SYS, B_INST, E_INST, E_SYS
import torch
import os
from tqdm import tqdm
from accelerate import Accelerator
import random


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


class ESPromptOutput:
    def __init__(self, dialog: List[str], situation: str, speakers: List[str], responses: Dict[str, str]):
        self.dialog = dialog
        self.situation = situation
        self.speakers = speakers
        self.responses = responses

    def to_dict(self):
        return {
            "dialog": self.dialog,
            "situation": self.situation,
            "speakers": self.speakers,
            "responses": self.responses
        }

    def __repr__(self):
        respones_str = ""
        for strategy, resp in self.responses.items():
            respones_str += f"{'*'*300}\n\n strategy: {strategy}\n\n response: {resp}\n\n{'*'*300}\n\n"
        return "_______________________".join([
            f"Situation: {self.situation}",
            f"Dialog: {self.dialog}",
            f"responses: {respones_str}"
        ])


def convert_to_llama2_chat_format(sys_msg: str, conversations: List[str]) -> str:
    """
        <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    """
    conv_0 = conversations[0]
    conversations = conversations[1:]

    result = f"<s>{B_INST} {B_SYS}{sys_msg}{E_SYS} {conv_0} {E_INST} "
    i = 0
    while i < len(conversations):
        ai_msg = conversations[i]
        human_msg = conversations[i+1]
        result += f"{ai_msg} </s><s>{B_INST} {human_msg} {E_INST} "
        i += 2

    return result


def get_model_and_tokenizer(model_name, cache_dir, load_in_4bit=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=True
        )
        # Copy the model to each device
        device_map = (
            {"": Accelerator().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    )

    return model, tokenizer


def get_continuation_prompt(conversation, model, tokenizer):
    dialog = conversation['dialog_history']
    speakers = conversation['prev_speakers']
    situation = conversation['situation']

    template = """You are a helpful, precise and accurate emotional support expert.\
 The user has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make your response short and to the point."""

    if speakers[0] == 'supporter':
        speakers = speakers[1:]
        dialog = dialog[1:]

    assert speakers[0] == 'seeker'
    assert speakers[-1] == 'seeker'
    responses = {}

    for strategy, desc in tqdm(modified_extes_support_strategies.items()):
        if random.random() > 0.3:
            continue
        sys_msg = template.format(situation=situation, cur_strategy=strategy, strategy_description=desc)
        prompt = convert_to_llama2_chat_format(sys_msg, dialog)

        input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)

        outputs = model.generate(input_ids, max_new_tokens=512, num_beams=5, num_return_sequences=1,
                                 do_sample=True, top_p=0.95, top_k=50, temperature=0.3, repetition_penalty=1.1,
                                 length_penalty=1.2)

        # outputs = model.generate(input_ids, max_new_tokens=512, num_return_sequences=1, repetition_penalty=1.1,
        #                          do_sample=False)

        response = outputs[0][len(input_ids[0]):]
        output_txt = tokenizer.decode(response, skip_special_tokens=True).strip()
        responses[strategy] = output_txt

    return ESPromptOutput(dialog=dialog, situation=situation, speakers=speakers, responses=responses)


def run(data_path='../original_data/train.json', min_turn=3, max_turn=12, model_path='meta-llama/Llama-2-7b-chat-hf',
        cache_dir=None, output_path='./outputs'):

    data = load_jsonl(data_path)
    data = [d for d in data if min_turn <= d['turn'] <= max_turn]
    model, tokenizer = get_model_and_tokenizer(model_path, cache_dir)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(output_path, exist_ok=True)

    for i, _ in enumerate(data):
        rand_id = random.randint(0, len(data))
        if os.path.exists(os.path.join(output_path, f'{rand_id}.json')):
            continue

        conversation = data[rand_id]
        generated_conts = get_continuation_prompt(conversation, model, tokenizer)

        with open(os.path.join(output_path, f'{rand_id}.json'), 'w') as f:
            json.dump(generated_conts.to_dict(), f)


if __name__ == '__main__':
    fire.Fire(run)

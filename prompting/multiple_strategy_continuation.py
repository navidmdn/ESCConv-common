import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict
import fire
from prompting.llama_prompt import modified_extes_support_strategies
from prompting.llama_prompt import B_SYS, B_INST, E_INST, E_SYS
import torch
import os
from tqdm import tqdm
from accelerate import Accelerator
import random

template1 = """You are a helpful, precise and accurate emotional support expert.\
The user has come to you with the following situation: "{situation}". continue the\
conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make your response short and to the point."""

template2 = """You are a helpful and caring friend.\
Your best friend has come to you with the following situation: "{situation}". continue the\
conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make your response short and to the point.\
 Do not provide additional info. only respond in one paragraph that satisfies {cur_strategy} strategy."""


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


def convert_to_llama2_chat_format_manually(sys_msg: str, conversations: List[str]) -> str:
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

def convert_to_llama2_chat_partial_conv_format(sys_msg: str, conversations: List[str], tokenizer, n_turns_as_conv=3) -> str:

    if n_turns_as_conv % 2 != 1:
        raise ValueError("n_turns_as_conv should be odd number")

    conv_messages = []
    for i in range(max(len(conversations)-n_turns_as_conv, 0), len(conversations)-1, 2):
        conv_messages.append({'role': 'user', 'content': conversations[i].strip()})
        conv_messages.append({'role': 'assistant', 'content': conversations[i+1].strip()})
    conv_messages.append({'role': 'user', 'content': conversations[-1].strip()})

    if len(conversations) > n_turns_as_conv:
        conversations = conversations[:-n_turns_as_conv]

    conv_history_str = "conversation history:\n"
    for i in range(0, len(conversations)-1, 2):
        conv_history_str += "user: " + conversations[i].strip() + "\n"
        conv_history_str += "assistant: " + conversations[i+1].strip() + "\n"

    sys_msg = f"{conv_history_str}\n{sys_msg.strip()}"
    messages = [{'role': 'system', 'content': sys_msg}]
    messages.extend(conv_messages)

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_prompt

def convert_to_llama2_chat_format(sys_msg: str, conversations: List[str], tokenizer) -> str:
    messages = [{'role': 'system', 'content': sys_msg}]
    for i in range(0, len(conversations)-1, 2):
        messages.append({'role': 'user', 'content': conversations[i]})
        messages.append({'role': 'assistant', 'content': conversations[i+1]})
    messages.append({'role': 'user', 'content': conversations[-1]})

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_prompt

def convert_to_llama2_llm_format(sys_msg: str, conversations: List[str], tokenizer) -> str:
    formatted_prompt = f"{tokenizer.bos_token}{sys_msg}\n\n"

    for i in range(0, len(conversations)-1, 2):
        formatted_prompt += f"seeker: {conversations[i].strip()}\n"
        formatted_prompt += f"supporter: {conversations[i+1].strip()}\n"
    formatted_prompt += f"seeker: {conversations[-1].strip()}\n"
    formatted_prompt += f"supporter: "
    return formatted_prompt


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


def get_continuation_prompt(conversation, model, tokenizer, model_type='llama'):
    dialog = conversation['dialog_history']
    speakers = conversation['prev_speakers']
    situation = conversation['situation']

    if speakers[0] == 'supporter':
        speakers = speakers[1:]
        dialog = dialog[1:]

    assert speakers[0] == 'seeker'
    assert speakers[-1] == 'seeker'
    responses = {}

    for strategy, desc in tqdm(modified_extes_support_strategies.items()):
        if random.random() > 0.3:
            continue
        sys_msg = template2.format(situation=situation, cur_strategy=strategy, strategy_description=desc)

        if model_type == 'llama' or model_type == 'mistral':
            prompt = convert_to_llama2_chat_partial_conv_format(sys_msg, dialog, tokenizer)

            input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)
            print("length of input_ids: ", len(input_ids[0]))
            print("prompt: ", prompt)
        elif model_type == 'mistral':
            pass
        else:
            raise ValueError(f"model_type should be one of ['llama', 'mistral'], but got {model_type}")

        outputs = model.generate(input_ids, max_new_tokens=512, num_beams=5, num_return_sequences=1,
                                 do_sample=True, top_p=0.95, top_k=50, temperature=0.8, repetition_penalty=1.1,
                                 length_penalty=1.2)

        response = outputs[0][len(input_ids[0]):]
        output_txt = tokenizer.decode(response, skip_special_tokens=True).strip()
        responses[strategy] = output_txt
        print("\n\nresponse: ", output_txt)

    res = ESPromptOutput(dialog=dialog, situation=situation, speakers=speakers, responses=responses)
    return res

def run(data_path='../original_data/train.json', min_turn=3, max_turn=12, model_path='meta-llama/Llama-2-7b-chat-hf',
        cache_dir=None, output_path='./outputs', load_in_4bit=True):

    data = load_jsonl(data_path)
    data = [d for d in data if min_turn <= d['turn'] <= max_turn]
    model, tokenizer = get_model_and_tokenizer(model_path, cache_dir, load_in_4bit)
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

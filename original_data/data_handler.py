from typing import Dict, Tuple, List
from transformers import PreTrainedTokenizer
import json
import torch

def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data


def write_json(file_path, data: List[Dict]):
    with open(file_path, 'w') as f:
        for line in data:
            f.write(json.dumps(line))
            f.write('\n')


def norm_strategy(strategy):
    norm_str = "-".join(strategy.split())
    return "@[" + norm_str + "]"


def get_strategy(file_path, norm=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = [d.replace('[', '').replace(']', '') for d in data]
    if norm:
        data = [norm_strategy(d) for d in data]
    return data


class InputPreprocessor:
    def __init__(
            self,
            preprocessor_type: str,
            tokenizer: PreTrainedTokenizer = None,
            max_source_length: int = None,
            max_target_length: int = None,
            supporter_token: str = '<supporter>',
            seeker_token: str = '<seeker>',
            sep_token: str = '<sep>',
            add_strategy_token: bool = False,
            strategy_list: List[str] = None,
    ):
        self.preprocessor_type = preprocessor_type
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.supporter_token = supporter_token
        self.seeker_token = seeker_token
        self.sep_token = sep_token
        self.add_strategy_token = add_strategy_token
        self.strategy_list = strategy_list

        # to keep the most recent conversation
        self.tokenizer.truncation_side = 'left'

        self.processing_functions = {
            'joint_strategy_utterance_generation': self.joint_strategy_utterance_generation,
            'strategy_generation': self.strategy_generation_tokenization,
            'utterance_generation_conditioned_on_strategy': self.utterance_generation_conditioned_on_strategy,
            'peft_clm_preprocessor': self.peft_clm_preprocessor,
            'peft_clm_formatting': self.peft_clm_formatting,
        }

        self.preprocess = self.processing_functions[preprocessor_type]

    def strategy_generation_tokenization(self, example):
        history = example['history']
        target = example['future_strategy']
        full_text = self.tokenizer.sep_token.join(history)

        inputs = self.tokenizer(full_text, add_special_tokens=True, max_length=self.max_source_length, truncation=True)
        labels = self.tokenizer(target, add_special_tokens=True, max_length=self.max_target_length, truncation=True)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids'],
        }

    def utterance_generation_conditioned_on_strategy(self, example):
        history = example['dialog_history']
        speakers = example['prev_speakers']
        target = example['response']
        cur_strategies = example['strategy']

        if isinstance(cur_strategies, str):
            cur_strategies = [norm_strategy(cur_strategies)]
        elif isinstance(cur_strategies, list):
            cur_strategies = [norm_strategy(s) for s in cur_strategies]
        else:
            raise Exception("strategy should be either str or list")

        assert "prev_strategies" in example
        prev_strategies = example['prev_strategies']

        prev_strategies = [[norm_strategy(s) for s in strategies] for strategies in prev_strategies]

        full_text = f"situation: {example['situation']} {self.sep_token} "
        assert len(history) == len(speakers) == len(prev_strategies)
        for speaker, utt, strategies in zip(speakers, history, prev_strategies):
            speaker_token = self.supporter_token if speaker == 'supporter' else self.seeker_token
            if self.add_strategy_token:
                strategy_tokens = "" if speaker == 'seeker' else "".join(strategies)
            else:
                strategy_tokens = ""

            full_text += f"{speaker_token}{strategy_tokens} {utt} "

        cur_strategies = "".join(cur_strategies)
        full_text += f"{self.supporter_token}{cur_strategies}"

        inputs = self.tokenizer(full_text, add_special_tokens=True, max_length=self.max_source_length, truncation=True)
        labels = self.tokenizer(target, add_special_tokens=True, max_length=self.max_target_length, truncation=True)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids'],
        }

    def joint_strategy_utterance_generation(self, example):
        history = example['history']
        target = example['response']
        full_text = self.tokenizer.bos_token + " ".join(history) + " <helper> "

        inputs = self.tokenizer(full_text, add_special_tokens=False, max_length=self.max_source_length, truncation=True)
        labels = self.tokenizer(target, add_special_tokens=True, max_length=self.max_target_length, truncation=True)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids'],
        }

    def peft_clm_preprocessor(self, example, window_size=2):
        history = example['dialog_history'][-window_size:]
        speakers = example['prev_speakers'][-window_size:]
        target = example['response']
        cur_strategies = example['strategy']

        if isinstance(cur_strategies, str):
            cur_strategies = [cur_strategies]
        elif not isinstance(cur_strategies, list):
            raise Exception("strategy should be either str or list")

        assert "prev_strategies" in example
        prev_strategies = example['prev_strategies'][-window_size:]

        full_text = f"situation: {example['situation']} conversation:\n"
        assert len(history) == len(speakers) == len(prev_strategies)
        for speaker, utt, strategies in zip(speakers, history, prev_strategies):
            speaker_token = self.supporter_token if speaker == 'supporter' else self.seeker_token
            full_text += f"{speaker_token}: {utt.strip()}\n"

        cur_strategies = "".join(cur_strategies)
        full_text += f"{self.supporter_token}: {cur_strategies} "

        inputs = self.tokenizer(full_text, add_special_tokens=True)
        labels = self.tokenizer(target, add_special_tokens=False)

        sample_input_ids = inputs["input_ids"]
        label_input_ids = labels["input_ids"] + [self.tokenizer.eos_token_id]
        inputs["input_ids"] = sample_input_ids + label_input_ids
        non_padded_inputs = inputs["input_ids"].copy()

        inputs["input_ids"] = [self.tokenizer.pad_token_id] * (self.max_source_length - len(inputs["input_ids"])) + \
                              non_padded_inputs
        labels["input_ids"] = [-100] * (len(inputs["input_ids"]) - len(label_input_ids)) + label_input_ids
        inputs["attention_mask"] = [0] * (len(inputs["input_ids"]) - len(non_padded_inputs)) + [1] * len(
            non_padded_inputs)

        assert len(inputs["input_ids"]) == len(labels["input_ids"]) == len(
            inputs["attention_mask"])

        return {
            'input_ids': torch.tensor(inputs['input_ids'][-self.max_source_length:]),
            'attention_mask': torch.tensor(inputs['attention_mask'][-self.max_source_length:]),
            'labels': torch.tensor(labels['input_ids'][-self.max_source_length:]),
            'full_text': full_text,
        }

    def peft_clm_formatting(self, examples, window_size=3):
        examples_dict = []
        for i in range(len(examples['dialog_history'])):
            examples_dict.append({
                'dialog_history': examples['dialog_history'][i],
                'prev_speakers': examples['prev_speakers'][i],
                'response': examples['response'][i],
                'strategy': examples['strategy'][i],
                'prev_strategies': examples['prev_strategies'][i],
                'situation': examples['situation'][i],
            })

        formatted_examples = []
        for example in examples_dict:
            history = example['dialog_history'][-window_size:]
            speakers = example['prev_speakers'][-window_size:]
            target = example['response']
            cur_strategies = example['strategy']

            if isinstance(cur_strategies, str):
                cur_strategies = [cur_strategies]
            elif not isinstance(cur_strategies, list):
                raise Exception("strategy should be either str or list")

            assert "prev_strategies" in example
            prev_strategies = example['prev_strategies'][-window_size:]

            full_text = f"situation: {example['situation']} conversation:\n"
            assert len(history) == len(speakers) == len(prev_strategies)
            for speaker, utt, strategies in zip(speakers, history, prev_strategies):
                speaker_token = self.supporter_token if speaker == 'supporter' else self.seeker_token
                full_text += f"{speaker_token}: {utt.strip()}\n"

            cur_strategies = "".join(cur_strategies)
            full_text += f"{self.supporter_token}: {cur_strategies} "
            full_text += target
            formatted_examples.append(full_text)

        return formatted_examples

    def joint_strategy_utterance_generation(self, example):
        history = example['history']
        target = example['response']
        full_text = self.tokenizer.bos_token + " ".join(history) + " <helper> "

        inputs = self.tokenizer(full_text, add_special_tokens=False, max_length=self.max_source_length, truncation=True)
        labels = self.tokenizer(target, add_special_tokens=True, max_length=self.max_target_length, truncation=True)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids'],
        }

import copy
import json
import os.path
import torch
from transformers import BartTokenizer
from typing import List, Dict
from fire import Fire
from transformers import AutoTokenizer, PreTrainedTokenizer



def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


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
    return "@["+norm_str+"]"


def get_strategy(file_path, norm=False):
    with open(file_path,'r', encoding='utf-8') as f:
        data = json.load(f)
    data = [d.replace('[','').replace(']','') for d in data]
    if norm:
        data = [norm_strategy(d) for d in data]
    print('strategy: ', data)

    return data


def _norm(x):
    return ' '.join(x.strip().split()).lower()


def construct_conversational_dataset(
        file_path: str,
        tokenizer: BartTokenizer,
        add_cause: bool = False,
        with_strategy: bool = False,
        load_from_cache: bool = True,
) -> str:

    split = 'train'
    if 'valid' in file_path:
        split = 'valid'
    elif 'test' in file_path:
        split = 'test'

    cached_preprocessed_file = file_path.replace(split, f"preprocessed_{split}")

    if os.path.exists(cached_preprocessed_file) and load_from_cache:
        print(f"loading preprocessed {split} dataset from {cached_preprocessed_file}")
        return cached_preprocessed_file

    data = load_json(file_path)
    print(f"parsing {file_path} file")
    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else " "
    print(f"setting {sep_token} as sep token")

    total_data = []
    is_train = 'train' in file_path

    # todo: check if required later:
    # if 'test' in file_path and self.model_type > 4:
    #     gt_strategy = read_pk('./final_data/test_extend_label.pk')
    #     if lookahead is not True:
    #         print("do not use lookahead! ")
    #         predict_strategy = read_pk('./final_data/wo_lookahead_predicted.pk')
    #     else:
    #         print('use lookahead! ')
    #         predict_strategy = read_pk('./final_data/multiesc_predicted_strategy.pk')
    #     print("acc: ", accuracy_score(gt_strategy, predict_strategy))

    predict_strategy_index = 0
    for case_example in data:
        dialog = case_example['dialog']
        dialog_len = len(dialog)
        emotion_type = case_example['emotion_type']
        problem_type = case_example['problem_type']
        situation = case_example['situation']
        tot_strategy = []
        for index, tmp_dic in enumerate(dialog):
            if tmp_dic['speaker'] == 'sys' and tmp_dic['strategy'] != "Others":
                # todo: doesnt it mess with data?
                tot_strategy.append(norm_strategy(tmp_dic['strategy']))

        # initial history is emotion_type + problem_type + situation
        history = [_norm(emotion_type) + sep_token + _norm(
            problem_type) + sep_token + _norm(situation)]

        tmp_strategy_list = []
        for index, tmp_dic in enumerate(dialog):
            text = _norm(tmp_dic['text'])
            if index == 0 and tmp_dic['speaker'] != 'sys':
                # todo: why prepend it to history?
                # handling when conversation starts with help seeker
                history[0] = text + sep_token + history[0]
                continue
            if tmp_dic['speaker'] == 'sys' and tmp_dic['strategy'] != "Others":
                # build one example from history up to this point of conversation
                tmp_strategy = norm_strategy(tmp_dic['strategy'])
                save_s = [x for x in tot_strategy[len(tmp_strategy_list):]].copy()
                assert len(save_s) > 0, print(tot_strategy, tmp_strategy_list)
                tmp_history = copy.deepcopy(history)
                response = text

                # todo: check if required later:
                # if with_strategy and self.model_type > 4:
                #     if 'test' in file_path:
                #         # tmp_history[-1] = tmp_history[-1] + self.sep_token + predict_strategy[predict_strategy_index]
                #         if self.model_type == 8:
                #             # response = predict_strategy[predict_strategy_index] + " " + text
                #             tmp_history.append(predict_strategy[predict_strategy_index])
                #         else:
                #             tmp_history[-1] = tmp_history[-1] + self.sep_token + predict_strategy[
                #                 predict_strategy_index]
                #         predict_strategy_index += 1
                #     else:
                #         if self.model_type == 8:
                #             tmp_history.append(tmp_strategy)
                #             # response = tmp_strategy + " " + text
                #         else:
                #             tmp_history[-1] = tmp_history[-1] + self.sep_token + tmp_strategy

                total_data.append({
                    "history": tmp_history,
                    "strategy": tmp_strategy,
                    "history_strategy": tmp_strategy_list,
                    "response": response,
                    "future_strategy": ' '.join(save_s),
                    # todo: what is the use case for stage of conversation?
                    "stage": 5 * index // dialog_len,
                })
                tmp_strategy_list.append(tmp_strategy)

            if tmp_dic['speaker'] == 'sys':
                # handling helper utterance with strategy
                tmp_strategy = norm_strategy(tmp_dic['strategy'])
                if with_strategy:
                    # add strategy as control code to the history text
                    tmp_sen = tmp_strategy + sep_token + text
                    history.append(tmp_sen)
                else:
                    history.append(text)
            else:
                # help seeker utterance
                utt = text

                if add_cause:
                    cause = tmp_dic['cause']
                    if cause is not None and len(cause) > 0:
                        utt = cause + sep_token + utt

                history.append(utt)

    # random_idx = random.randint(0, len(total_data) - 1)
    # print(f'printing training example {random_idx}:')
    # print(total_data[random_idx])

    # todo: limiting just for test
    write_json(cached_preprocessed_file, total_data[:50])
    return cached_preprocessed_file


class InputPreprocessor:
    def __init__(
            self,
            preprocessor_type: str,
            tokenizer: PreTrainedTokenizer = None,
            max_source_length: int = None,
            max_target_length: int = None,
    ):
        self.preprocessor_type = preprocessor_type
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        processing_functions = {
            'joint_strategy_utterance_generation': self.sequence_only_strategy_generation_tokenization,
        }

        self.preprocess = processing_functions[preprocessor_type]

    def sequence_only_strategy_generation_tokenization(self, example):
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


# class CustomDataCollator(DataCollator):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#
#     def sequence_only_strategy_generation_collator(self, batch):
#         pass

def main(
    base_file_path: str,
    tokenizer_name_or_path: str = 'facebook/bart-base',
    add_cause: bool = False,
    with_strategy: bool = False,
):
    splits = ['train', 'valid', 'test']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    for split in splits:
        file_path = os.path.join(base_file_path, f"{split}.json")
        construct_conversational_dataset(
            file_path,
            tokenizer,
            add_cause=add_cause,
            with_strategy=with_strategy,
            load_from_cache=False,
        )


if __name__ == "__main__":
    Fire(main)
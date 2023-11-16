import pandas as pd
import fire
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from original_data.data_handler import get_strategy, load_json
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import re
import evaluate as evallib
import torch
import nltk
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from original_data.data_handler import InputPreprocessor
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

# need to run onece
# nltk.download('punkt')


def calculate_belu(responses, targets):
    b4 = []
    b3 = []
    b2 = []
    stemmer = nltk.stem.PorterStemmer()

    for resp, target in zip(responses, targets):

        ref = nltk.tokenize.word_tokenize(target)
        hyp = nltk.tokenize.word_tokenize(resp)

        b4.append(sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25)))
        b3.append(sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33)))
        b2.append(sentence_bleu([ref], hyp, weights=(0.5, 0.5)))

    return b2, b3, b4


def calculate_rouge(responses, targets):
    rouge = evallib.load("rouge")

    rouge2 = []
    rougeL = []
    rouge1 = []
    rougeLsum = []

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in responses]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in targets]

    for pred, label in zip(preds, labels):
        result = rouge.compute(predictions=[pred], references=[labels], use_stemmer=True)
        rouge1.append(result['rouge1'])
        rouge2.append(result['rouge2'])
        rougeL.append(result['rougeL'])
        rougeLsum.append(result['rougeLsum'])

    return rouge1, rouge2, rougeL, rougeLsum

def calculate_evaluation_metrics(responses, targets):
    b4_sum = b3_sum = b2_sum = 0
    metric_res = {}
    stemmer = nltk.stem.PorterStemmer()

    for resp, target in zip(responses, targets):
        b4_sum = b3_sum = b2_sum = 0

        ref = nltk.tokenize.word_tokenize(target)
        hyp = nltk.tokenize.word_tokenize(resp)

        ref = [stemmer.stem(w) for w in ref]
        hyp = [stemmer.stem(w) for w in hyp]

        b4_sum += sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25))
        b3_sum += sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33))
        b2_sum += sentence_bleu([ref], hyp, weights=(0.5, 0.5))

    bleu_dict = {
        'B4': b4_sum / len(targets),
        'B3': b3_sum / len(targets),
        'B2': b2_sum / len(targets),
    }
    metric_res.update(bleu_dict)

    rouge = evallib.load("rouge")

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in responses]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in targets]

    result = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}

    metric_res.update(result)
    return metric_res


def evaluate_strategy_conditioned_response_generator(model_path, strategy_path, test_data_path,
                                                     cache_dir=None, experiment_name='default', batch_size=16,
                                                     sample_size=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.truncation_side = 'left'
    # note that special tokens should already be added to the tokenizer
    strategy_list = get_strategy(strategy_path, norm=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)

    test_dataset = load_dataset(
        'json',
        data_files={'test': test_data_path},
        cache_dir=cache_dir,
        # for testing
    )['test']

    if sample_size is not None:
        test_dataset = test_dataset.shuffle(seed=42).select(range(sample_size))

    preprocessor = InputPreprocessor(
        preprocessor_type='utterance_generation_conditioned_on_strategy',
        tokenizer=tokenizer,
        max_source_length=300,
        max_target_length=128,
        add_strategy_token=False,
        strategy_list=strategy_list,
    )
    preprocessor_func = preprocessor.preprocess
    test_dataset = test_dataset.map(preprocessor_func, num_proc=4, remove_columns=test_dataset.column_names)

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    responses = []
    targets = []

    print("generating responses...")
    for i, batch in tqdm(enumerate(dataloader), total=len(test_dataset)//batch_size):

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128,
                                 num_beams=3, repetition_penalty=1.1,)
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)
        targets.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

        responses.extend(res)

    metrics = {}
    b2, b3, b4 = calculate_belu(responses, targets)
    rouge1, rouge2, rougeL, rougeLsum = calculate_rouge(responses, targets)

    df_metrics = {
        'B2': b2,
        'B3': b3,
        'B4': b4,
        'Rouge1': rouge1,
        'Rouge2': rouge2,
        'RougeL': rougeL,
        'RougeLsum': rougeLsum,
        'response': responses,
        'target': targets,
    }

    df = pd.DataFrame(df_metrics)
    df.to_csv(f'{experiment_name}.csv', index=False, header=True)

    # metrics.update(df.mean().to_dict())
    return metrics


def evaluate_joint_strategy_and_utterance_generator(model_path, base_model, test_data_path):
    # setting truncation side to keep last tokens of conversation
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.truncation_side = 'left'
    print("raw vocab size: ", len(tokenizer))
    # retrieve strategy list for token control coding
    strategy_list = get_strategy('new_strategy.json', norm=True)
    tokenizer.add_special_tokens({"sep_token": "<sep>"})
    tokenizer.add_tokens(strategy_list)
    print("new vocab size: ", len(tokenizer))

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    test_data = load_json(test_data_path)

    strategy_dict = {}
    # leave 0 for out of vocabulary strategy
    for s in strategy_list:
        strategy_dict[s] = len(strategy_dict) + 1

    responses = []
    targets = []
    turn_numbers = []

    strategy_preds = []
    strategy_labels = []

    print("generating responses...")
    for i, entry in tqdm(enumerate(test_data)):
        if i == 100:
            break
        history = entry['history']
        target = entry['response']
        full_text = tokenizer.bos_token + " ".join(history) + " <helper> "
        turn_numbers.append(len(re.findall(r"<helper>", full_text)))

        input_ids = tokenizer(full_text, add_special_tokens=False, truncation=True,
                              return_tensors='pt', max_length=512).input_ids

        outputs = model.generate(input_ids, max_length=256, num_beams=5, early_stopping=True)
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)

        pred_strategy = re.findall(r'^@\[.+\]', res)
        label_strategy = re.findall(r'^@\[.+\]', target)

        if len(label_strategy) != 0:
            label_strategy = label_strategy[0]
            strategy_labels.append(strategy_dict[label_strategy])

            if len(pred_strategy) == 0:
                strategy_preds.append(0)
            if pred_strategy[0] in strategy_dict:
                strategy_preds.append(strategy_dict[pred_strategy[0]])
            else:
                strategy_preds.append(0)

        utterance = re.sub(r'^@\[.+\]', '', res).strip()
        target = re.sub(r'^@\[.+\]', '', target).strip()

        responses.append(utterance)
        targets.append(target)

    metrics = {}
    b2, b3, b4 = calculate_belu(responses, targets)
    rouge1, rouge2, rougeL, rougeLsum = calculate_rouge(responses, targets)

    df_metrics = {
        'B2': b2,
        'B3': b3,
        'B4': b4,
        'Rouge1': rouge1,
        'Rouge2': rouge2,
        'RougeL': rougeL,
        'RougeLsum': rougeLsum,
        'turn': turn_numbers
    }

    df = pd.DataFrame(df_metrics)
    df.to_csv('metrics.csv', index=False, header=True)

    metrics.update(df.mean().to_dict())
    strategy_pred_acc = accuracy_score(strategy_labels, strategy_preds)
    metrics['strategy_pred_acc'] = strategy_pred_acc
    return metrics


def evaluate(
        model_path,
        test_data_path,
        base_model='facebook/bart-base',
        model_type='joint_strategy_utt',
        strategy_path=None,
        cache_dir=None,
        experiment_name='default',
        batch_size=32,
        sample_size=None,
):

    if model_type == 'joint_strategy_utt':
        results = evaluate_joint_strategy_and_utterance_generator(model_path, base_model, test_data_path)
    elif model_type == 'utterance_generation_conditioned_on_strategy':
        assert strategy_path is not None
        # assuming the tokenizer is saved in model_path
        results = evaluate_strategy_conditioned_response_generator(model_path, strategy_path, test_data_path, cache_dir,
                                                                   experiment_name, batch_size, sample_size)
    else:
        raise NotImplementedError

    print(results)


if __name__ == '__main__':
    fire.Fire(evaluate)
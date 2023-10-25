import pandas as pd
import fire
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data.data_handler import get_strategy, load_json
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import re
import evaluate as evallib
import nltk
from sklearn.metrics import accuracy_score

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

def evaluate(model_path, test_data_path, base_model='facebook/bart-base', model_type='joint_strategy_utt'):
    if model_type == 'joint_strategy_utt':
        results = evaluate_joint_strategy_and_utterance_generator(model_path, base_model, test_data_path)
    else:
        raise NotImplementedError

    print(results)

if __name__ == '__main__':
    fire.Fire(evaluate)
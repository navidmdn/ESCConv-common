import json
import fire
import torch
from transformers.models.roberta import RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm

def preprocess_train_examples_with_ExTES_strategies(examples_batch, strategy_classifier_model,
                                                    strategy_classifier_tokenizer, id2strategy, device, max_length=128):
    inputs = []
    for example in examples_batch:
        cur_inp = ""
        for utt, speaker in zip(example['dialog_history'], example['prev_speakers']):
            cur_inp += f"<{speaker}> {utt} "
        cur_inp += f"<supporter> {example['response']}"
        inputs.append(cur_inp)

    inputs = strategy_classifier_tokenizer(inputs, padding="max_length", max_length=max_length,
                                           truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        strategies = strategy_classifier_model(**inputs).logits
    strategies = torch.argmax(strategies, dim=1).tolist()

    modified_strategies = []
    for pred_strategy, example in zip(strategies, examples_batch):
        example_strategy = set(example['strategy'])
        if len(example_strategy) == 1 and list(example_strategy)[0] == 'Others':
            strategy = 'Others'
        else:
            strategy = id2strategy[pred_strategy]
        modified_strategies.append(strategy)

    return modified_strategies


def run(model_path, data_path, output_path, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    classifier = classifier.to(device)
    classifier.eval()

    tokenizer.truncation_side = "left"

    label2id = classifier.config.label2id
    id2label = {v: k for k, v in label2id.items()}

    with open(data_path, "r") as f:
        examples = json.load(f)

    annotated = []
    idx = 0
    while idx < len(examples):
        strategies = preprocess_train_examples_with_ExTES_strategies(examples[idx:idx+batch_size], classifier,
                                                                     tokenizer, id2label, device=device)
        for example, strategy in tqdm(zip(examples[idx:idx+batch_size], strategies)):
            example['strategy'] = strategy
            annotated.append(example)

        idx += batch_size

    with open(output_path, "w") as f:
        for example in annotated:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    fire.Fire(run)
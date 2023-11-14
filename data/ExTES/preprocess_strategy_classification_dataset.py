import fire
import json
from typing import Dict, Union, List, Optional
import os
import spacy
from tqdm import tqdm


VALID_STRATEGIES = [
    "Reflective Statements",
    "Clarification",
    "Emotional Validation",
    "Empathetic Statements",
    "Affirmation",
    "Offer Hope",
    "Avoid Judgment and Criticism",
    "Suggest Options",
    "Collaborative Planning",
    "Provide Different Perspectives",
    "Reframe Negative Thoughts",
    "Share Information",
    "Normalize Experiences",
    "Promote Self-Care Practices",
    "Stress Management",
    "Others"
]
VALID_STRATEGIES = [s.lower() for s in VALID_STRATEGIES]

def preprocess_conversation(conversation: Dict, conv_window=1) -> Dict:
    history = conversation['dialog_history'][-conv_window:]
    prev_speakers = conversation['prev_speakers'][-conv_window:]
    response = conversation['response']
    strategy = conversation['strategy'][0]

    assert len(strategy) > 0

    full_text = ""
    for utt, speaker in zip(history, prev_speakers):
        full_text += f"<{speaker}> {utt} "

    full_text += f"<supporter> {response}"

    assert strategy in VALID_STRATEGIES, "strategy not found: " + strategy

    return {
        "sentence": full_text,
        "strategy": strategy
    }


def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def preprocess(
        data_dir: str = ".",
        output_prefix: str = "sfe_",
):

    train = load_jsonl(os.path.join(data_dir, "train.json"))
    valid = load_jsonl(os.path.join(data_dir, "valid.json"))
    test = load_jsonl(os.path.join(data_dir, "test.json"))

    def build_strategy_classification_dataset(data: List[Dict]):
        print("initial size of dataset: ", len(data))

        examples = []
        for conversation in data:
            ex = preprocess_conversation(conversation)
            if ex is not None:
                examples.append(ex)

        print("total instances after preprocessing: ", len(examples))
        return examples

    for split, data_split in zip(['train', 'valid', 'test'], [train, valid, test]):
        output_file_path = os.path.join(data_dir, f"{output_prefix}{split}.json")
        examples = build_strategy_classification_dataset(data_split)
        with open(output_file_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example))
                f.write("\n")


if __name__ == '__main__':
    fire.Fire(preprocess)

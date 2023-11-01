import fire
import json
from typing import Dict, Tuple, List


VALID_STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
]

def preprocess_conversation(conversation: Dict) -> Dict:
    strategy = conversation['strategy'][0]
    dialog_history = conversation["dialog_history"]
    input_txt = ""
    for speaker, content in zip(conversation['prev_speakers'], dialog_history):
        input_txt += f"<{speaker}>{content} "

    input_txt += f"<supporter>{conversation['response']}"
    return {
        "context": input_txt,
        "strategy": strategy,
    }

def preprocess(
        data_path: str = "test.json",
        output_prefix: str = "sfe_",
):
    with open(data_path, 'r') as f:
        data = json.load(f)

    print("total initial instances: ", len(data))
    def preprocess_and_save(data, output_path):
        examples = []
        for conversation in data:
            # for now skipping multi strategy and irrelevant ones
            if len(conversation['strategy']) > 1 or conversation['strategy'][0] not in VALID_STRATEGIES:
                continue
            examples.append(preprocess_conversation(conversation))
        print("total instances after preprocessing: ", len(examples))
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example))
                f.write("\n")

    split = 'train' if 'train' in data_path else 'valid' if 'valid' in data_path else 'test'
    output_file_path = data_path.replace(split, f"{output_prefix}{split}")
    preprocess_and_save(data, output_file_path)


if __name__ == '__main__':
    fire.Fire(preprocess)

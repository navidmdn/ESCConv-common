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
    "Others"
]


def preprocess_conversation(conversation: Dict, conversation_window=1) -> Dict:
    strategy = conversation['strategy'][0]
    dialog_history = conversation["dialog_history"][-conversation_window:]
    speakers = conversation['prev_speakers'][-conversation_window:]
    input_txt = ""
    for speaker, content in zip(speakers, dialog_history):
        input_txt += f"<{speaker}> {content} "

    input_txt += f"<supporter> {conversation['response']}"
    return {
        "sentence": input_txt,
        "strategy": strategy,
    }

def preprocess(
        data_path: str = "train.json",
        output_prefix: str = "sfe_",
):
    data = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))

    print("total initial instances: ", len(data))
    def preprocess_and_save(data, output_path):
        examples = []
        for conversation in data:
            # for now skipping multi strategy and irrelevant ones
            if len(conversation['strategy']) > 1 or conversation['strategy'][0] == "":
                continue
            assert conversation['strategy'][0] in VALID_STRATEGIES
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

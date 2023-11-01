import fire
import json
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split

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
]

def preprocess_conversation(conversation: Dict) -> List[Dict]:
    history = conversation['content']
    processed_text = ""
    results = []

    n_turn = 0
    for turn in history:
        if 'AI' in turn:
            speaker = 'supporter'
            n_turn += 1
        elif 'User' in turn:
            speaker = 'seeker'
        else:
            raise ValueError("Unknown speaker: ", turn)
        utt = turn['AI' if speaker == 'supporter' else 'User']
        processed_text += f"<{speaker}>{utt} "
        if n_turn > 3 and speaker == 'supporter' and "AI Strategy" in turn and turn['AI Strategy'] in VALID_STRATEGIES:
            strategy = turn['AI Strategy']
            results.append({'context': processed_text, 'strategy': strategy})

    return results

def preprocess(
        data_path: str = "ExTES.json",
        output_prefix: str = "sfe_",
):
    with open(data_path, 'r') as f:
        data = json.load(f)

    print("total initial instances: ", len(data))
    examples = []
    for conversation in data:
        # skip incomplete conversations
        incomplete = False
        for turn in conversation['content']:
            if 'AI' not in turn and 'User' not in turn:
                incomplete = True
        if incomplete:
            continue
        examples.extend(preprocess_conversation(conversation))
    print("total instances after preprocessing: ", len(examples))

    train, test = train_test_split(data, test_size=0.1, random_state=42, shuffle=True)
    train, valid = train_test_split(train, test_size=0.1, random_state=42, shuffle=True)

    for split, data_split in zip(['train', 'valid', 'test'], [train, valid, test]):
        output_file_path = data_path.replace('ExTES', f"{output_prefix}{split}")
        with open(output_file_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example))
                f.write("\n")


if __name__ == '__main__':
    fire.Fire(preprocess)

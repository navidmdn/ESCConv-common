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
    "Others"
]

def preprocess_conversation(conversation: Dict) -> List[Dict]:
    history = conversation['content']
    processed_text = ""
    results = []
    all_speakers = []
    n_turn = 0
    for turn in history:
        ai_strategy = ""
        if "AI" in turn and "User" in turn:
            user_utt = turn["User"]
            ai_utt = turn["AI"]
            ai_strategy = turn["AI Strategy"] if "AI Strategy" in turn else "Others"
            n_turn += 1

            # checking whose turn it is first
            if len(all_speakers) == 0 or all_speakers[-1] == 'supporter':
                all_speakers.append('seeker')
                all_speakers.append('supporter')
                processed_text += f"<seeker>{user_utt} "
                processed_text += f"<supporter>{ai_utt} "
                speaker = 'supporter'
            elif all_speakers[-1] == 'seeker':
                all_speakers.append('supporter')
                all_speakers.append('seeker')
                processed_text += f"<supporter>{ai_utt} "
                processed_text += f"<seeker>{user_utt} "
                speaker = 'seeker'
            else:
                raise Exception("unhandled case")
        else:
            if 'AI' in turn:
                speaker = 'supporter'
                n_turn += 1
            elif 'User' in turn:
                speaker = 'seeker'
            else:
                raise ValueError("Unknown speaker: ", turn)
            utt = turn['AI' if speaker == 'supporter' else 'User']
            processed_text += f"<{speaker}>{utt} "
            all_speakers.append(speaker)
            if "AI Strategy" in turn:
                ai_strategy = turn["AI Strategy"]

        if n_turn > 3 and speaker == 'supporter' and ai_strategy != "" and ai_strategy in VALID_STRATEGIES:
            results.append({'context': processed_text, 'strategy': ai_strategy})

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

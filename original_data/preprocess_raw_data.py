import fire
import json
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


def decompose_conversation(conversation: Dict, starting_turn: int, turn_by_turn=True) -> List[Dict]:

    history = conversation['dialog']
    emotion_type = conversation['emotion_type']
    problem_type = conversation['problem_type']
    situation = conversation['situation']

    all_turns = []
    all_speakers = []
    all_strategies = []
    decomposed_examples = []

    for turn_obj in history:
        speaker = turn_obj['speaker']
        content = turn_obj['content']
        annotation = turn_obj['annotation']

        if 'strategy' in annotation:
            strategy = annotation['strategy']
        elif speaker == 'supporter':
            strategy = 'Others'
        else:
            strategy = ''

        all_turns.append(content)
        all_speakers.append(speaker)
        all_strategies.append(strategy)

    if turn_by_turn:
        concat_turns = [all_turns[0]]
        concat_speakers = [all_speakers[0]]
        concat_strategies = [[all_strategies[0]]]

        for i in range(1, len(all_turns)):
            prev_speaker = all_speakers[i-1]
            cur_speaker = all_speakers[i]

            if cur_speaker != prev_speaker:
                concat_turns.append(all_turns[i])
                concat_speakers.append(all_speakers[i])
                concat_strategies.append([all_strategies[i]])
                continue

            concat_turns[-1] += ' ' + all_turns[i]

            prev_strategy = concat_strategies[-1][-1]
            cur_strategy = all_strategies[i]

            if cur_strategy != prev_strategy:
                concat_strategies[-1].append(cur_strategy)

        all_turns = concat_turns
        all_speakers = concat_speakers
        all_strategies = concat_strategies

    conv_so_far = []
    speakers_so_far = []
    strategies_so_far = []

    max_turn = len([x for x in all_speakers if x == 'supporter']) - 1

    turn = 0
    for i, (turn_content, speaker, strategies) in enumerate(zip(all_turns, all_speakers, all_strategies)):
        # don't count as a turn if supporter starts the conversation
        if speaker == 'supporter' and i > 0:
            turn += 1

        if turn == max_turn:
            break

        if speaker == 'supporter' and turn >= starting_turn:
            decomposed_examples.append({
                'emotion_type': emotion_type,
                'problem_type': problem_type,
                'situation': situation,
                'dialog_history': conv_so_far.copy(),
                'prev_speakers': speakers_so_far.copy(),
                'prev_strategies': strategies_so_far.copy(),
                'strategy': strategies,
                'response': turn_content,
                'turn': turn,
            })

        conv_so_far.append(turn_content)
        speakers_so_far.append(speaker)
        strategies_so_far.append(strategies)

    return decomposed_examples


def preprocess(
        data_path: str = "ESConv.json",
        output_dir: str = ".",
        starting_turn: int = 3,

):
    with open(data_path, 'r') as f:
        data = json.load(f)

    conversations = []

    for conversation in data:
        conversations.extend(decompose_conversation(conversation, starting_turn=starting_turn))

    # split data to train, val, test
    # do a 0.6, 0.2, 0.2 split
    train, test = train_test_split(conversations, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    train, valid = train_test_split(train, test_size=0.25, random_state=RANDOM_SEED)

    with open(f'{output_dir}/train.json', 'w') as f:
        json.dump(train, f)

    with open(f'{output_dir}/valid.json', 'w') as f:
        json.dump(valid, f)

    with open(f'{output_dir}/test.json', 'w') as f:
        json.dump(test, f)


if __name__ == '__main__':
    fire.Fire(preprocess)

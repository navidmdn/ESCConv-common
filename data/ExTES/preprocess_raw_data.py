import fire
import json
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_md")


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
strategy_embs = [nlp(s) for s in VALID_STRATEGIES]

RANDOM_SEED = 42

def find_strategy(strategy: str, valid_strategies: List[str], strategy_embs) -> str:
    if strategy in valid_strategies:
        return strategy
    # matching with available strategies
    if strategy.lower() in valid_strategies:
        return strategy.lower()

    #todo: fix this to use all data
    # removing any ambiguous strategies
    return

    if strategy.lower() == 'validation':
        return "emotional validation"

    sim = 0
    best_strategy = None
    for strategy_text, s in zip(valid_strategies, strategy_embs):
        cur_sim = s.similarity(nlp(strategy.lower()))
        if cur_sim > sim:
            sim = cur_sim
            best_strategy = strategy_text

    if sim < 0.8:
        # print("strategy not found: ", strategy)
        return

    print(f"{strategy} -> {best_strategy} - score: {sim}")
    return best_strategy

def decompose_conversation(conversation: Dict, starting_turn: int, turn_by_turn=True) -> List[Dict]:
    history = conversation['content']
    emotion_type = ""
    problem_type = ""
    situation = ""

    if "scene" in conversation:
        problem_type = conversation['scene']

    if "description" in conversation:
        situation = conversation['description']

    all_turns = []
    all_speakers = []
    all_strategies = []
    decomposed_examples = []

    for turn_obj in history:
        if "AI" in turn_obj and "User" in turn_obj:
            user_utt = turn_obj["User"]
            ai_utt = turn_obj["AI"]

            ai_strategy = "others"
            if "AI Strategy" in turn_obj:
                ai_strategy = turn_obj["AI Strategy"]
                if ai_strategy is None or len(ai_strategy) == 0:
                    ai_strategy = "others"

                ai_strategy = find_strategy(ai_strategy, VALID_STRATEGIES, strategy_embs)
                if ai_strategy is None:
                    return []

            # checking whose turn it is first
            if len(all_speakers) == 0 or all_speakers[-1] == 'supporter':
                all_turns.append(user_utt)
                all_speakers.append('seeker')
                all_strategies.append('')
                all_turns.append(ai_utt)
                all_speakers.append('supporter')
                all_strategies.append(ai_strategy)
            elif all_speakers[-1] == 'seeker':
                all_turns.append(ai_utt)
                all_speakers.append('supporter')
                all_strategies.append(ai_strategy)
                all_turns.append(user_utt)
                all_speakers.append('seeker')
                all_strategies.append('')
            else:
                raise Exception("unhandled case")
        else:
            if 'AI' in turn_obj:
                speaker = 'supporter'
            elif 'User' in turn_obj:
                speaker = 'seeker'
            else:
                # skip this conversation totally
                return []

            content = turn_obj["AI" if speaker == 'supporter' else "User"]

            # seeker always gets empty strategy
            # supporter gets strategy if it's available otherwise gets Others as strategy
            strategy = "others"
            if 'AI Strategy' in turn_obj and speaker == 'supporter':
                if len(turn_obj["AI Strategy"]) > 0:
                    strategy = turn_obj['AI Strategy']

            strategy = find_strategy(strategy, VALID_STRATEGIES, strategy_embs)
            if strategy is None:
                return []

            if speaker == 'seeker':
                strategy = ""

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

    for ss in all_strategies:
        for s in ss:
            if s is None:
                # corrupted data
                return []

    conv_so_far = []
    speakers_so_far = []
    strategies_so_far = []

    max_turn = len([x for x in all_speakers if x == 'supporter']) - 1

    turn = 0
    assert len(all_turns) == len(all_speakers) == len(all_strategies)
    for i, (turn_content, speaker, strategies) in enumerate(zip(all_turns, all_speakers, all_strategies)):
        # don't count as a turn if supporter starts the conversation

        if isinstance(turn_content, list) or isinstance(turn_content, dict):
            return []

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
        data_path: str = "ExTES.json",
        output_dir: str = ".",
        starting_turn: int = 1,

):
    with open(data_path, 'r') as f:
        data = json.load(f)

    # split data to train, val, test
    # do a 0.6, 0.2, 0.2 split
    train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    train, valid = train_test_split(train, test_size=0.25, random_state=RANDOM_SEED)
    
    def preprocess_and_save(split_data, split):
        conversations = []
        corrupted = 0
        for conversation in tqdm(split_data):
            # todo: only add starting turn for training
            if split == 'train':
                decomposed = decompose_conversation(conversation, starting_turn=starting_turn)
                if len(decomposed) == 0:
                    corrupted += 1
                conversations.extend(decomposed)
            else:
                decomposed = decompose_conversation(conversation, starting_turn=1)
                if len(decomposed) == 0:
                    corrupted += 1
                conversations.extend(decomposed)
        print(f"corrupted conversations: {corrupted}/{len(split_data)}")
        with open(f'{output_dir}/{split}.json', 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')

    for split, split_data in zip(['train', 'valid', 'test'], [train, valid, test]):
        preprocess_and_save(split_data, split)


if __name__ == '__main__':
    fire.Fire(preprocess)

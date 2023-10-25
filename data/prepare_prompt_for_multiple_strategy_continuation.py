import json
import random
from data_handler import load_json, write_json


with open("../ESConv.json", 'r') as f:
    data = json.load(f)

strategies = {
    "Question": """In emotional support counseling, asking open-ended questions is crucial. These questions encourage the 
    individual to express their thoughts and feelings. They can be used to explore the person's emotions, triggers, 
    or concerns. For instance, you might ask, "What is causing you to feel this way?""",
    "Restatement or Paraphrasing": """This technique involves repeating or rephrasing what the person has said to show 
    that you're actively listening and trying to understand. For instance, if someone says, "I'm really stressed about 
    work," you can respond with, "It sounds like work is causing you a lot of stress." """,
    "Reflection of Feelings": """Reflecting feelings involves acknowledging and validating the person's emotions. This 
    helps the individual feel heard and understood. If someone expresses sadness, you might respond with, "I can see that 
    you're feeling very sad about this situation." """,
    "Self-disclosure": """Carefully sharing your own relevant experiences or feelings can help build rapport and trust. 
    However, it should be done in a way that doesn't shift the focus from the person seeking support. For example, 
    "I've also felt overwhelmed at times, and it's completely normal." """,
    "Affirmation and Reassurance": """Providing affirmations and reassurance helps boost the individual's self-esteem and 
    confidence. You might say, "You're doing a great job by seeking help, and I believe you have the strength to overcome 
    this." """,
    "Providing Suggestions or Information": """Sometimes, individuals seek guidance or information. When appropriate, 
    offer suggestions or information that can help them cope or find resources. For example, "Have you considered trying 
    mindfulness meditation to manage your stress? It can be quite effective." """,
}
strategies = {k.lower(): v for k, v in strategies.items()}


def prepare_prompt(data_json):
    emotion_type = data_json['emotion_type']
    problem_type = data_json['problem_type']
    situation = data_json['situation']

    history = data_json['dialog']

    prompt = "You are an emotional support counselor and here is the situation:\n\
    the help seeker has come to you with {emotion_type} from {problem_type}. The situation is as follows:\n\
    {situation}\n\n{cont_strategy} strategy: {cont_strategy_def}\nHere is a conversation between help seeker and counselor. At each turn, \
    the counselor is trying to help the help seeker by specifically following the mentioned strategy.\n\n"

    dial_hist = []

    n_supporter_turns = len([x for x in history if x['speaker'] == 'supporter'])
    assert n_supporter_turns > 3
    random_break_point = random.randint(3, n_supporter_turns-1)

    supporter_turn = 0
    for turn in history:
        content = turn['content'].strip()
        dial_hist.append(content)
        speaker = turn['speaker']
        if speaker == 'supporter':
            annt = turn['annotation']

            if 'strategy' not in annt:
                strategy = 'chitchat'
            else:
                strategy = annt['strategy'].lower()
            supporter_turn += 1
            if supporter_turn == random_break_point:
                cont_strategy = random.choice(list(strategies.keys()))
                while cont_strategy == strategy:
                    cont_strategy = random.choice(list(strategies.keys()))
                prompt += f"<strategy: {cont_strategy}>\n{speaker}: "
                break
            prompt += f"<strategy: {strategy}>\n{speaker}: {content}\n\n"
        else:
            prompt += f"{speaker}: {content}\n\n"

    prompt = prompt.format(emotion_type=emotion_type, problem_type=problem_type, situation=situation,
                           cont_strategy=cont_strategy, cont_strategy_def=strategies[cont_strategy],)
    return prompt, dial_hist, cont_strategy


prompts = []
for entry in data:
    prompt, dial_hist, strategy = prepare_prompt(entry)
    prompts.append({'prompt': prompt, 'dialog_history': dial_hist, 'strategy': strategy})

write_json('./train_prompt.json', prompts)

i = random.randint(0, len(prompts)-1)
print(prompts[i]['prompt'])
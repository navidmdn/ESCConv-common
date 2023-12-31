import json

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import random
import json
app = Flask(__name__)

# ESConve data path
# data_path = '../../original_data/train_ann_extes.json'

# ExTES data path
data_path = '../../original_data/test.json'
# model_response_path = None # If you want to visualize model responses, set this to the path of the model response file

model_response_path = '../../results/llama2chat.csv'
# model_response_path = '../../results/conv_prefix_clm.csv'

gens = []
if model_response_path is not None:
    if model_response_path.endswith('.csv'):
        df = pd.read_csv(model_response_path)
        gens = df['response'].tolist()
    elif model_response_path.endswith('.json'):
        with open(model_response_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                gens.append(obj['response'])
    else:
        raise ValueError('model_response_path must be either .csv or .json')


# Sample dataset of conversations
# This should be replaced with your actual dataset

conversations = []
with open(data_path, 'r') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        obj['index'] = len(conversations) + 1
        messages = []
        history = obj['dialog_history']
        speakers = obj['prev_speakers']
        prev_strategies = obj['prev_strategies']
        for speaker, text, strategy in zip(speakers, history,prev_strategies):
            strategy = strategy[0] if isinstance(strategy, list) else strategy
            messages.append({"speaker": speaker, "text": text, "strategy": strategy})
        resp_strategy = obj['strategy'][0] if isinstance(obj['strategy'], list) else obj['strategy']
        messages.append({'speaker': 'supporter', 'text': obj['response'], 'strategy': resp_strategy})
        if model_response_path is not None:
            messages[-1].update({"gen_response": gens[i]})

        metadata = {
            'emotion_type': obj['emotion_type'],
            'problem_type': obj['problem_type'],
            'situation': obj['situation'],
        }
        conversations.append({"index": obj['index'], "messages": messages, "metadata": metadata})


# here is a template to follow
# conversations = [
#     {"index": 1, "messages": [{"speaker": "User", "text": "Hello, how are you?"}, {"speaker": "Bot", "text": "I am fine, thank you!"}]},
#     {"index": 2, "messages": [{"speaker": "User", "text": "What's the weather like?"}, {"speaker": "Bot", "text": "It's sunny today!"}]}
# ]

@app.route('/')
def index():
    return render_template('index.html')  # A simple form to input conversation index


@app.route('/random_conversation')
def random_conversation():
    random_index = random.randint(1, len(conversations))  # Assuming index starts at 1
    return redirect(url_for('conversation', index=random_index))


@app.route('/conversation/<int:index>', methods=['GET'])
def conversation(index):
    conversation = next((conv for conv in conversations if conv['index'] == index), None)
    if conversation:
        next_index = index + 1 if index + 1 <= len(conversations) else None
        return render_template('conversation.html', conversation=conversation['messages'], index=index,
                               next_index=next_index, metadata=conversation['metadata'])
    else:
        return f"No conversation found with index: {index}"


if __name__ == '__main__':
    app.run(debug=True)
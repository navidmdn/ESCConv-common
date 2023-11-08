import json

from flask import Flask, render_template, request

app = Flask(__name__)


data_path = '../../original_data/train_ann_extes.json'
# Sample dataset of conversations
# This should be replaced with your actual dataset

conversations = []
with open(data_path, 'r') as f:
    for line in f:
        obj = json.loads(line)
        obj['index'] = len(conversations) + 1
        messages = []
        history = obj['dialog_history']
        speakers = obj['prev_speakers']
        for speaker, text in zip(speakers, history):
            messages.append({"speaker": speaker, "text": text})
        messages.append({'speaker': 'supporter', 'text': obj['response']})
        conversations.append({"index": obj['index'], "messages": messages})


# here is a template to follow
# conversations = [
#     {"index": 1, "messages": [{"speaker": "User", "text": "Hello, how are you?"}, {"speaker": "Bot", "text": "I am fine, thank you!"}]},
#     {"index": 2, "messages": [{"speaker": "User", "text": "What's the weather like?"}, {"speaker": "Bot", "text": "It's sunny today!"}]}
# ]

@app.route('/')
def index():
    return render_template('index.html')  # A simple form to input conversation index

@app.route('/conversation', methods=['POST'])
def conversation():
    index = int(request.form.get('index'))
    conversation = next((conv for conv in conversations if conv['index'] == index), None)
    if conversation:
        return render_template('conversation.html', conversation=conversation['messages'])
    else:
        return f"No conversation found with index: {index}"

if __name__ == '__main__':
    app.run(debug=True)
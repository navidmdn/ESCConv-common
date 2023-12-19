import json

result = []
with open('ESConv.json', 'r') as f:
    data = json.load(f)
    for i, conv in enumerate(data):
        cur_conv_str = ""
        dialog = conv['dialog']
        for turn in dialog:
            cur_conv_str += turn['speaker'] + ": " + turn['content'].strip() + "\n"
        result.append(cur_conv_str)


with open('plain_conv.txt', 'w') as f:
    f.write('\n\n********\n********\n\n'.join(result))

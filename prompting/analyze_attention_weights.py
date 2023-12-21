from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def aggregate_attention(attn):
    # todo: experiment with different aggregation strategies
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            # todo: shouldn't we take current token's row instead?
            attns_per_head[-1][1:],
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def decode(tokens):
    '''Turn tokens into text with mapping index'''
    full_text = ''
    chunks = []
    for i, token in enumerate(tokens):
        text = tokenizer.decode(token)
        full_text += text
        chunks.append(text)
    return full_text, chunks


def get_completion(prompt):
    '''Get full text, token mapping, and attention matrix for a completion'''
    # tokens = tokenizer.encode(prompt, return_tensors="pt")
    tokens = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)

    outputs = model.generate(
        tokens,
        max_new_tokens=50,
        output_attentions=True,
        return_dict_in_generate=True,
        early_stopping=True,
        length_penalty=-1
    )
    sequences = outputs.sequences
    attn_m = heterogenous_stack([
                                    torch.tensor([
                                        1 if i == j else 0
                                        for j, token in enumerate(tokens[0])
                                    ])
                                    for i, token in enumerate(tokens[0])
                                ] + list(map(aggregate_attention, outputs.attentions)))
    decoded, tokenized = decode(sequences[0])
    return decoded, tokenized, attn_m


def show_matrix(xs, tokens):
    for x in xs:
        line = ''
        for y in x:
            line += '{:.4f}\t'.format(float(y))
        print(line)
    tokens_line = ''
    for token in tokens:
        token = repr(token)[1:-1][:6]
        if len(token) >= 6:
            partial_space = ""
        else:
            partial_space = " " * (6 - len(token))
        # print("token: ", token)
        # print("len(token): ", len(token))
        # print("partial_space: ", len(partial_space))
        tokens_line += token + partial_space + '\t'
    print(tokens_line)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    config = model.config
    print(config)

    prompt = 'given the info navid is a phd student, where does navid work?'
    decoded, tokenized, attn_m = get_completion(prompt)
    show_matrix(attn_m, tokenized)

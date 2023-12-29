import torch
from typing import List
from transformers import LlamaForCausalLM, PreTrainedTokenizer


def default_aggregate_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.cpu().squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)

        # in case the generation is produced by beam search, also average over beams
        # todo: need to use attention indices to filter instead of just averaging
        if len(attns_per_head.shape) == 3:
            attns_per_head = attns_per_head.mean(dim=0)

        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:],
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def find_sublist_start(larger: List[int], smaller: List[int]):
    if len(smaller) > len(larger):
        raise ValueError("sublist cannot be longer than reference list")

    if len(smaller) == 0 or len(larger) == 0:
        raise ValueError("cannot find sublist in empty list")

    for i in range(len(larger) - len(smaller) + 1):
        if larger[i:i+len(smaller)] == smaller:
            return i

    return -1


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def get_average_attention_over_sequence(aggregated_attention, token_ids: torch.Tensor, sequence: str,
                                        tokenizer: PreTrainedTokenizer):
    # check if sequence is available in token_ids
    seq_token_ids = tokenizer(sequence, add_special_tokens=False, return_tensors='pt')['input_ids'][0].tolist()

    # to avoid initial token mismatch due to space or newline
    seq_token_ids = seq_token_ids[1:]

    prompt_token_ids = token_ids[:len(token_ids)-len(aggregated_attention)]

    #for test:
    prompt = tokenizer.decode(prompt_token_ids)
    # print("prompt: ", prompt)
    # print("strategy: ", sequence)


    attn_m = heterogenous_stack([
        torch.tensor([
            1 if i == j else 0
            for j, token in enumerate(prompt_token_ids)
        ])
        for i, token in enumerate(prompt_token_ids)
    ] + aggregated_attention)

    beg_idx = find_sublist_start(token_ids.tolist(), seq_token_ids)

    if beg_idx == -1:
        raise ValueError(f"sequence {seq_token_ids} not found in reference token_ids: {token_ids}")

    avg_attn_score = attn_m[-len(aggregated_attention):, beg_idx:beg_idx+len(seq_token_ids)].mean().item()
    return avg_attn_score


if __name__ == '__main__':

    import pickle
    from transformers import AutoTokenizer

    with open('../../llm_attention_viz/llm_attention_viz/data/1002_attentions.pkl', 'rb') as f:
        ex = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

    strategies = list(ex.keys())

    strategy, (tokens, attentions) = strategies[-1], ex[strategies[-1]]
    strategy = f'"{strategy}"'
    print(get_average_attention_over_sequence(attentions, tokens, sequence=strategy, tokenizer=tokenizer))




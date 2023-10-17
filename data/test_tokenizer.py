from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", )
tok = BartTokenizer.from_pretrained("facebook/bart-base")
example_english_phrase = "ok</s>ok"

print("special tokens:")
print(tok.special_tokens_map)

batch = tok(example_english_phrase, return_tensors="pt")
print(batch)

decoded_inputs = tok.batch_decode(batch["input_ids"], skip_special_tokens=False)
print("decoded inputs:")
print(decoded_inputs)

generated_ids = model.generate(batch["input_ids"])

print("decoded:")
print(tok.batch_decode(generated_ids, skip_special_tokens=False))
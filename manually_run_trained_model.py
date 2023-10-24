from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from data.data_handler import get_strategy

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
print("raw vocab size: ", len(tokenizer))
# retrieve strategy list for token control coding
strategy_list = get_strategy('new_strategy.json', norm=True)
tokenizer.add_special_tokens({"sep_token": "<sep>"})
tokenizer.add_tokens(strategy_list)
print("new vocab size: ", len(tokenizer))

model = AutoModelForSeq2SeqLM.from_pretrained('output/bart-base/checkpoint-3000')
# print(model)

def get_response(input_text):
    full_text = tokenizer.bos_token + input_text + tokenizer.sep_token

    input_ids = tokenizer(full_text, add_special_tokens=False, max_length=500, truncation=True, return_tensors='pt').input_ids
    print("input shape:", input_ids.shape)
    outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(res)

get_response("\
hello good afternoon .<sep>anxiety<sep>job crisis<sep>i am on short term disability and i am afraid i will lose my job\
 if i don't go back soon.<sep>@[Question] hi , good afternoon .<sep>i ' m feeling anxious that i am going to lose\
 my job . .<sep>@[Reflection-of-feelings] losing a job is always anxious .<sep>i hope i don ' t .<sep>@[Question] why\
 do you think you will lose your job ?<sep>i am on short term disability and i am not ready to go back to work yet but\
 i do not have any job protection .")

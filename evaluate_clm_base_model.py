
from transformers import AutoModelForCausalLM, AutoTokenizer
from original_data.data_handler import get_strategy, InputPreprocessor
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import fire
from accelerate import Accelerator
from transformers import default_data_collator, BitsAndBytesConfig


def evaluate(model_name_or_path, test_file_path, strategy_file=None, cache_dir=None, seq_len=256):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=False
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    model.eval()

    test_dataset = load_dataset(
        'json',
        data_files={'test': test_file_path},
        cache_dir=cache_dir,
    )['test']

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # strategy_list = get_strategy(strategy_file, norm=False)
    preprocessor = InputPreprocessor(
        preprocessor_type='peft_clm_preprocessor_for_dialogpt',
        tokenizer=tokenizer,
        max_source_length=seq_len,
        max_target_length=seq_len,
        add_strategy_token=False,
    )
    preprocessor_func = preprocessor.preprocess
    raw_datasets = test_dataset.map(preprocessor_func, num_proc=4, remove_columns=test_dataset.column_names)

    #todo: for testing purpose
    raw_datasets = raw_datasets.select(range(100))
    prompts = raw_datasets['prompt']
    raw_datasets.remove_columns(['prompt'])

    data_loader = DataLoader(raw_datasets, collate_fn=default_data_collator, batch_size=4, shuffle=False)
    print(f"Stopping generation with {tokenizer.eos_token_id} token id: {tokenizer.eos_token}")

    responses = []
    for batch in data_loader:
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items()}

            for input_id, attention_mask, label in zip(inputs['input_ids'], inputs['attention_mask'], inputs['labels']):
                input_id = input_id[label == -100].unsqueeze(0)
                attention_mask = attention_mask[label == -100].unsqueeze(0)
                # print(input_id, attention_mask, label)
                output = model.generate(
                    input_ids=input_id, attention_mask=attention_mask, max_new_tokens=50,
                    eos_token_id=[tokenizer.eos_token_id], num_beams=3,
                    num_return_sequences=1,
                )
                print(output)
                output_decoded = tokenizer.decode(output[0].detach().cpu().numpy(), skip_special_tokens=False)
                print(output_decoded)
                responses.append(output_decoded)

    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}")
        print(f"Response: {response.split(prompt)[-1]}")
        print("*" * 80)


if __name__ == '__main__':
    fire.Fire(evaluate)

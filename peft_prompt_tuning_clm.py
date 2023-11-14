# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
import random
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit, TaskType, MultitaskPromptTuningConfig, MultitaskPromptTuningInit
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,\
    HfArgumentParser, TrainingArguments
from original_data.data_handler import get_strategy, InputPreprocessor
from transformers import Trainer
from peft import get_peft_model
from transformers import default_data_collator


tqdm.pandas()


# Define and parse arguments. 2
@dataclass
class ScriptArguments:
    train_file: Optional[str] = field(metadata={"help": "training file path"})
    validation_file: Optional[str] = field(metadata={"help": "validation file path"})
    strategy_file: Optional[str] = field(metadata={"help": "strategy file path"})
    preprocess_type: Optional[str] = field(default="peft_clm_preprocessor", metadata={"help": "preprocess type"})
    peft_type: Optional[str] = field(default="prompt_tuning", metadata={"help": "peft type"})

    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=256, metadata={"help": "Input sequence length"})
    eval_steps: Optional[int] = field(default=25, metadata={"help": "the number of evaluation steps"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "the cache directory"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=10, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=50, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = (
            {"": Accelerator().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        cache_dir=script_args.cache_dir,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_auth_token=script_args.use_auth_token,
    )

    # when running locally on cpu
    # model = AutoModelForCausalLM.from_pretrained(script_args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,)

    # Step 2: Load the dataset

    data_files = {}
    if script_args.train_file is not None:
        data_files["train"] = script_args.train_file
        extension = script_args.train_file.split(".")[-1]
    if script_args.validation_file is not None:
        data_files["validation"] = script_args.validation_file
        extension = script_args.validation_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=script_args.cache_dir,
    )
    columns = raw_datasets["train"].column_names

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    strategy_list = get_strategy(script_args.strategy_file, norm=False)
    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        raise Exception("The vocab size shouldn't have changed!")

    preprocessor = InputPreprocessor(
        preprocessor_type=script_args.preprocess_type,
        tokenizer=tokenizer,
        seeker_token="seeker",
        supporter_token="supporter",
        max_source_length=script_args.seq_length,
        max_target_length=script_args.seq_length,
        strategy_list=strategy_list,
        add_strategy_token=False,
    )
    preprocessor_func = preprocessor.preprocess

    raw_datasets = raw_datasets.map(preprocessor_func, load_from_cache_file=False, num_proc=4, remove_columns=columns)
    raw_datasets.remove_columns(["prompt"])

    train_dataset = raw_datasets["train"]
    valid_dataset = raw_datasets["validation"]

    print("A sample of train dataset: ")
    idx = random.randint(0, len(train_dataset))
    input_ids = train_dataset[idx]['input_ids']
    label_ids = train_dataset[idx]['labels']
    attention_mask = train_dataset[idx]['attention_mask']
    # decode input_ids
    print("input_ids: ", input_ids)
    print("attention_mask: ", attention_mask)
    print("label ids: ", label_ids)
    print("input_ids decoded: ", tokenizer.decode(input_ids, skip_special_tokens=False))

    label_ids = np.where(np.array(label_ids) == -100, tokenizer.pad_token_id, label_ids)
    print("label_ids decoded: ", tokenizer.decode(label_ids, skip_special_tokens=False))

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        # TODO: uncomment that on the next release
        # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
    )

    # Step 4: Define the LoraConfig
    if script_args.use_peft:
        print("using peft type: ", script_args.peft_type)

        if script_args.peft_type == 'prompt_tuning':
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=25,
                prompt_tuning_init_text="Continue the following conversation assuming that you are an emotional supporter",
                tokenizer_name_or_path=script_args.model_name,
            )
        elif script_args.peft_type == 'multitask_prompt_tuning':
            peft_config = MultitaskPromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=MultitaskPromptTuningInit.TEXT,
                num_virtual_tokens=25,
                prompt_tuning_init_text="Continue the following conversation assuming that you are an emotional supporter",
                num_tasks=len(strategy_list),
                tokenizer_name_or_path=script_args.model_name,
            )
        else:
            raise NotImplementedError()

        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.print_trainable_parameters()
    # Step 5: Define the Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    # Step 6: Save the model
    trainer.save_model(script_args.output_dir)


if __name__ == "__main__":
    main()
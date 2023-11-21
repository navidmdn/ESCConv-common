from trainers.llama_prompt import *
from dataclasses import dataclass, field
from typing import Optional
import random
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, \
    HfArgumentParser, TrainingArguments, AutoConfig
from transformers import Trainer
from peft import get_peft_model
from modeling.modeling_llama import LlamaForCausalLMWithConditionalPrompt
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM


def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


class LLamaPreprocessingForCLMWithConversationPrefix:
    def __init__(self, tokenizer, max_length, max_history_length=30):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.max_history_length = max_history_length
        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = True


    def truncate_or_pad(self, tokens, max_length, pad_token_id):
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]
        else:
            tokens = [pad_token_id] * (max_length - len(tokens)) + tokens
        return tokens

    def preprocess_for_llama_chat(self, example):

        assert self.tokenizer.add_bos_token and self.tokenizer.add_eos_token

        history = example['dialog_history']
        speakers = example['prev_speakers']
        dialog: List[Message] = []
        cur_strategies = example['strategy'][-1]
        strategy_description = ""
        if cur_strategies in strategy_descriptions:
            strategy_description = f"({strategy_descriptions[cur_strategies]})"
        response = Message(role="assistant", content=example['response'])

        system_msg = f"""You are a helpful emotional support expert.\
 The user has come to you with the following situation: "{example['situation']}". continue the\
 conversation for one turn using "{cur_strategies}"{strategy_description} strategy."""

        # model only supports starting with user
        if speakers[0] == 'supporter':
            speakers = speakers[1:]
            history = history[1:]

        dialog.append(Message(role="system", content=system_msg))
        for speaker, utt in zip(speakers, history):
            if speaker == 'supporter':
                dialog.append(Message(role="assistant", content=utt))
            elif speaker == 'seeker':
                dialog.append(Message(role="user", content=utt))
            else:
                raise Exception("speaker should be either 'supporter' or 'seeker'")

        assert (
                dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"

        if dialog[0]["role"] == "system":
            dialog = [
                         {
                             "role": dialog[1]["role"],
                             "content": B_SYS
                                        + dialog[0]["content"]
                                        + E_SYS
                                        + dialog[1]["content"],
                         }
                     ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens: List[int] = sum(
            [
                self.tokenizer(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    #todo: check if bos and eos are added both
                    add_special_tokens=True,
                )["input_ids"]
                for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
            ],
            [],
        )

        dialog_tokens += self.tokenizer(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST} ",
            #todo: check bos added and eos not added
            add_special_tokens=True,
        )["input_ids"][:-1]


        label_tokens = self.tokenizer(
            f"{(response['content']).strip()}",
            #todo: check if eos added but not bos
            add_special_tokens=True,
        )["input_ids"][1:]

        full_input_tokens = dialog_tokens + label_tokens
        label_tokens = [-100] * len(dialog_tokens) + label_tokens
        attention_mask = [1] * len(full_input_tokens)

        #todo: make sure you are adding pad token to the tokenizer
        full_input_tokens = self.truncate_or_pad(full_input_tokens, self.max_length, self.tokenizer.pad_token_id)
        label_tokens = self.truncate_or_pad(label_tokens, self.max_length, -100)
        attention_mask = self.truncate_or_pad(attention_mask, self.max_length, 0)

        history_encodings = torch.tensor(example['encoded_history']) # k x history_emb_dim
        history_mask = [1] * history_encodings.shape[0]
        history_mask = self.truncate_or_pad(history_mask, self.max_history_length, 0)

        if history_encodings.shape[0] < self.max_history_length:
            history_encodings_padding = torch.zeros((self.max_history_length - history_encodings.shape[0],
                                                     history_encodings.shape[1]))
            history_encodings = torch.concat([history_encodings_padding, history_encodings], dim=0)
        else:
            print(f"truncating conversation history > {self.max_history_length}")
            history_encodings = history_encodings[-self.max_history_length:]

        return {
            'input_ids': torch.tensor(full_input_tokens),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(label_tokens),
            'prompt_ids': torch.tensor(dialog_tokens),
            'conversation_history_encodings': history_encodings,
            'conversation_history_mask': torch.tensor(history_mask)
        }

    def collate_batch(self, batch):
        input_ids = [example['input_ids'] for example in batch]
        attention_mask = [example['attention_mask'] for example in batch]
        labels = [example['labels'] for example in batch]
        history_encodings = [example['conversation_history_encodings'] for example in batch]
        history_mask = [example['conversation_history_mask'] for example in batch]

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        history_encodings = torch.tensor(history_encodings)
        history_mask = torch.tensor(history_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'conversation_history_encodings': history_encodings,
            'conversation_history_mask': history_mask
        }


@dataclass
class ScriptArguments:
    train_file: Optional[str] = field(metadata={"help": "training file path"})
    validation_file: Optional[str] = field(metadata={"help": "validation file path"})
    peft_type: Optional[str] = field(default="lora", metadata={"help": "peft type"})
    conv_hidden_size: Optional[int] = field(default=768, metadata={"help": "the hidden size of the conversation encoder"})
    prefix_fanout: Optional[int] = field(default=2, metadata={"help": "the fanout of the prefix"})

    model_name: Optional[str] = field(default="nickypro/tinyllama-15M", metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=200, metadata={"help": "Input sequence length"})
    eval_steps: Optional[int] = field(default=25, metadata={"help": "the number of evaluation steps"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "the cache directory"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "trian model in fp16 precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
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


    #todo: for testing
    # tokenizer = AutoTokenizer.from_pretrained('pretrained_models/')
    # config = LlamaConfig(num_attention_heads=4, num_hidden_layers=2, num_key_value_heads=4, hidden_size=128)
    # base_model = LlamaForCausalLM(config)

    config = AutoConfig.from_pretrained(script_args.model_name, cache_dir=script_args.cache_dir)

    base_model = LlamaForCausalLM.from_pretrained(
        script_args.model_name,
        config=config,
        quantization_config=quantization_config,
        device_map=device_map,
        cache_dir=script_args.cache_dir,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_auth_token=script_args.use_auth_token,
    )
    freeze_params(base_model)
    base_model.eval()

    print("base model's memory footprint: ", base_model.get_memory_footprint())

    model = LlamaForCausalLMWithConditionalPrompt(prefix_fanout=script_args.prefix_fanout,
                                                  conv_hidden_size=script_args.conv_hidden_size,
                                                  base_model=base_model)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, cache_dir=script_args.cache_dir)

    print_trainable_parameters(model)


    ###############


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

    # todo: just for test
    # raw_datasets['train'] = raw_datasets['train'].select(range(100))
    # raw_datasets['validation'] = raw_datasets['validation'].select(range(100))

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        raise Exception("The vocab size shouldn't have changed!")

    data_processor = LLamaPreprocessingForCLMWithConversationPrefix(tokenizer, script_args.seq_length)
    raw_datasets = raw_datasets.map(data_processor.preprocess_for_llama_chat,
                                    load_from_cache_file=True, num_proc=4, remove_columns=columns)

    train_dataset = raw_datasets["train"]
    valid_dataset = raw_datasets["validation"]

    print("A sample of train dataset: ")
    idx = random.randint(0, len(train_dataset))
    input_ids = train_dataset[idx]['input_ids']
    label_ids = train_dataset[idx]['labels']
    attention_mask = train_dataset[idx]['attention_mask']
    prompt_ids = train_dataset[idx]['prompt_ids']
    # decode input_ids
    print("input_ids: ", input_ids)
    print("attention_mask: ", attention_mask)
    print("label ids: ", label_ids)
    print("input_ids decoded: ", tokenizer.decode(input_ids, skip_special_tokens=False))
    print("prompt_ids: ", prompt_ids)
    print("prompt_ids decoded: ", tokenizer.decode(prompt_ids, skip_special_tokens=False))

    label_ids = np.where(np.array(label_ids) == -100, tokenizer.pad_token_id, label_ids)
    print("label_ids decoded: ", tokenizer.decode(label_ids, skip_special_tokens=False))

    raw_datasets.remove_columns(["prompt_ids"])

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
        save_safetensors=False,
        evaluation_strategy="steps",
        fp16=script_args.fp16,
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
        if script_args.peft_type == 'lora':
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=script_args.peft_lora_r,
                lora_alpha=script_args.peft_lora_alpha,
                lora_dropout=0.1,
                bias="all"
            )
        else:
            raise NotImplementedError()

        model = get_peft_model(model, peft_config)

    # model.print_trainable_parameters()
    # Step 5: Define the Trainer

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_processor.collate_batch,
    )

    trainer.train()

    # Step 6: Save the model
    trainer.save_model(script_args.output_dir)


if __name__ == "__main__":
    main()

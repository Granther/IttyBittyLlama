from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, get_scheduler
from huggingface_hub import HfApi, notebook_login
from datasets import load_dataset
from peft import LoraConfig, LoraModel, get_peft_model
from timm.scheduler import CosineLRScheduler
from accelerate import Accelerator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os



model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_id = "BAAI/Infinity-Instruct"

model_kwargs = dict(
    use_cache=False,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="sequential",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 2048
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
model = get_peft_model(model, lora_conf)

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

trainable_params = format(count_trainable_parameters(model), ",")

epochs = 1
per_dev_batch_size = 1
gradient_accumulation_steps = 30
dtype = torch.bfloat16
learning_rate = 1e-5

raw_dataset = load_dataset(dataset_id, "0625", split="train")

def apply_chat_template(example, tokenizer):
    convo = example['conversations']
    for dic in convo:
        dic['role'] = dic.pop('from')
        dic['content'] = dic.pop('value')
        if dic['role'] == 'gpt':
            dic['role'] = 'assistant'
        elif dic['role'] == 'human':
            dic['role'] = 'user'

    example['text'] = tokenizer.apply_chat_template(convo, tokenize=True, add_generation_prompt=False, truncation=True)
    return example

train_dataset = raw_dataset.select(range(100000))
test_dataset = raw_dataset.select(range(300))
column_names = list(train_dataset.features)

processed_train_dataset = train_dataset.map(
    apply_chat_template,
    # batched=True,
    # batch_size=20,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
)

processed_test_dataset = test_dataset.map(
    apply_chat_template,
    # batched=True,
    # batch_size=20,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataloader = torch.utils.data.DataLoader( # 
    processed_train_dataset['text'],
    batch_size=per_dev_batch_size,
    shuffle=True,
    collate_fn=data_collator
)

test_dataloader = torch.utils.data.DataLoader(
    processed_test_dataset['text'],
    batch_size=per_dev_batch_size,
    shuffle=True,
    collate_fn=data_collator
)

global_step = 0
num_training_steps = epochs * len(train_dataloader)
warmup_ratio = 0.1
warmup_steps = 1000
#warmup_steps = int(warmup_ratio * num_training_steps)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
cross_entropy = nn.CrossEntropyLoss()

scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

acc = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

if acc.is_main_process:
    wandb.init(
        project="tiny-llama-instruct",
    
        config={
        "learning_rate": learning_rate,
        "dataset": dataset_id,
        "batch_size": per_dev_batch_size,
        "lora_r": lora_conf.r,
        "lora_alpha": lora_conf.lora_alpha,
        "lora_dropout": lora_conf.lora_dropout,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "trainable_params": trainable_params,
        "num_training_steps": num_training_steps, 
        "model_name": "TinyLlama"
        }
    )   

optimizer, scheduler, train_dataloader, tokenizer, model, scheduler = acc.prepare(optimizer, scheduler, train_dataloader, tokenizer, model, scheduler)

def calc_metrics():
    model.eval()
    for batch in test_dataloader:
        pred = model(**batch)
        loss = pred.loss

        if acc.is_main_process:
            perplexity = torch.exp(loss)
            wandb.log({"eval_loss": loss.item(), "eval_perplexity": perplexity})

    model.train()

device = acc.device

model.train()
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        
        # outputs = model(**batch)
        # loss = outputs.loss
        
        # acc.backward(loss)
        # wandb.log({"loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr'], "perplexity": perplexity})

        with acc.accumulate(model):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            acc.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if acc.is_main_process:
                perplexity = torch.exp(loss)
                wandb.log({"loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr'], "perplexity": perplexity})
                
            global_step += 1
        
        # if (step + 1) % gradient_accumulation_steps == 0:
        #     optimizer.step()
        #     scheduler.step()
        #     optimizer.zero_grad()
        #     global_step += 1
    
        if (step + 1) % 100 == 0 and acc.is_main_process:
            print(f"Loss: {loss.item()}")
            
        if (step + 1) % 400 == 0:
            calc_metrics()

        if global_step > num_training_steps:
            break

    if global_step > num_training_steps:
        break

if acc.is_main_process:
    wandb.finish()

    save_path = os.path.join("checkpoint_instruct_2", f"step_{global_step}")
    model.module.save_pretrained(save_path)

    print("Saved model")
import os
import torch
from datasets import load_dataset, Split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

from cornell_movie_dialog import CornellMovieDialog

os.environ['WANDB_DISABLED']="true"

print(torch.backends.mps.is_available())
print(torch.backends.mkl.is_available())
print(torch.backends.cpu.get_cpu_capability())
print(torch.backends.cuda.is_built())
print(torch.backends.cudnn.is_available())
print(torch.backends.mkldnn.is_available())
print(torch.backends.openmp.is_available())
print(torch.backends.quantized.engine)

cornell_movie_dialog = CornellMovieDialog(data_dir="cornell movie-dialogs corpus")



# # Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# # New instruction dataset
# cornell_dataset = "cornell_movie_dialog"

# # Fine-tuned model
new_model = "llama-2-7b-chat-movie"

# dataset = load_dataset(cornell_dataset, split="train")
cornell_movie_dialog.download_and_prepare(output_dir="cornell_movie_dialog")
dataset = cornell_movie_dialog.as_dataset(split=Split(name="train"))
# print(dataset)

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)


mps_device = torch.device("cpu")
x = torch.ones(1, device=mps_device)
print (x)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"cpu":0})

model.config.use_cache = False
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to=["tensorboard"]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

# trainer.model.save_pretrained(new_model)
# trainer.tokenizer.save_pretrained(new_model)


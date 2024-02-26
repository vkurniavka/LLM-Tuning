import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from datasets import Split

from cornell_movie_dialog import CornellMovieDialog, DialogDataset

model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with the actual model name
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the model is in CPU mode
model.to("cpu")

print('load dataset')
cornell_movie_dialog = CornellMovieDialog(data_dir="cornell movie-dialogs corpus")
cornell_movie_dialog.download_and_prepare(output_dir="cornell_movie_dialog")
full_dataset = cornell_movie_dialog.as_dataset(split=Split(name="train"))

tokenizer = PreTrainedTokenizerFast(tokenizer_file='model/tokenizer.json')

dataset = DialogDataset(dataset=full_dataset, tokenizer=tokenizer)
batch_size = 1

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print('dataset loaded successfully')


# Fine-tuning the model
print('prepare model for train')
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print('start epoch')
for epoch in range(3):  # Number of epochs
    for batch in train_loader:
        print('optimizer.zero_grad')
        optimizer.zero_grad()
        print('batch to cpu')
        input_ids = batch.to('cpu')
        print('model inputs')
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        print('loss backword')
        loss.backward()
        print('step')
        optimizer.step()

    print(f"Epoch {epoch} completed")




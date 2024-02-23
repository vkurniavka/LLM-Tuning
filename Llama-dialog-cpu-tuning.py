import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from datasets import Split

from cornell_movie_dialog import CornellMovieDialog

model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with the actual model name
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the model is in CPU mode
model.to("cpu")

cornell_movie_dialog = CornellMovieDialog(data_dir="cornell movie-dialogs corpus")
cornell_movie_dialog.download_and_prepare(output_dir="cornell_movie_dialog")
dataset = cornell_movie_dialog.as_dataset(split=Split(name="train"))

batch_size = 1

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Fine-tuning the model
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):  # Number of epochs
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch.to('cpu')
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} completed")




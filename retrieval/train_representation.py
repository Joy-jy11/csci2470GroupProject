import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import IPython.display as display
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from transformers import get_scheduler
from info_nce import InfoNCE
from model import RepresentationNN, Projection_Model
from util import TextImageDataset, collate_fn, retrieval_eval


folder_path = 'Your data path for sampled data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Define model class and functions


# training function, optimized with InfoNCE Loss and AdamW
def train(
        model: nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader,
        optimizer: torch.optim, loss_fn: nn.Module, lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        num_epochs: int = 10, negative: int | None = None, accumulation_steps: int = 5
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # Move input data to the device
            text_inputs = {k: v.to(device) for k, v in batch["input_text"].items()}
            image_inputs = {k: v.to(device) for k, v in batch["input_images"].items()}
            text_representation, image_representation = model(text_inputs, image_inputs)
            # Compute loss with or without negatives
            if negative is not None:
                negative_key = []
                for i in range(text_representation.size(0)):
                    negative_candidate = [j for j in range(image_representation.size(0)) if j != i]
                    keys = np.random.choice(negative_candidate, negative)
                    negative_key.append(image_representation[keys])
                negative_key = torch.stack(negative_key)
                loss = loss_fn(text_representation, image_representation, negative_key)
            else:
                loss = loss_fn(text_representation, image_representation)
            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()  # Accumulate gradients
            # Update weights and reset gradients every `accumulation_steps`
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps  # Scale back to full batch loss for reporting
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss/len(train_dataloader):.4f}")
        evaluate(model, valid_dataloader, loss_fn, negative=negative)

def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module = None, validation: bool = True, negative: int | None = None):
    model.eval()
    if validation:
        with torch.no_grad():
            total_loss = 0.0
            # for batch in tqdm(dataloader):
            for batch in tqdm(dataloader):
                # Normalize embeddings for cosine similarity
                # text_inputs, image_inputs = text_inputs, image_inputs
                text_inputs = {k: v.to(device) for k, v in batch["input_text"].items()}
                image_inputs = {k: v.to(device) for k, v in batch["input_images"].items()}
                text_representation, image_representation = model(text_inputs, image_inputs)
                # labels = torch.ones(text_representation.size(0)).to(device)
                if negative is not None:
                    negative_key = []
                    for i in range(text_representation.size(0)):
                        negative_candidate = [j for j in range(image_representation.size(0)) if j != i]
                        keys = np.random.choice(negative_candidate, negative)
                        negative_key.append(image_representation[keys])
                    negative_key = torch.stack(negative_key)
                    loss = loss_fn(text_representation, image_representation, negative_key)
                else:
                    loss = loss_fn(text_representation, image_representation)
                total_loss += loss.item()
            print(f"Validatation Loss: {total_loss/len(dataloader):.4f}")
    # save the computed representation if this is to inference on testing data
    else:
        text_features = []
        image_features = []
        with torch.no_grad():
            total_loss = 0.0
            # for batch in tqdm(dataloader):
            for batch in tqdm(dataloader):
                # text_inputs, image_inputs = text_inputs.to(device), image_inputs.to(device)
                text_inputs = {k: v.to(device) for k, v in batch["input_text"].items()}
                image_inputs = {k: v.to(device) for k, v in batch["input_images"].items()}
                text_representation, image_representation = model(text_inputs, image_inputs)
                text_features.append(text_representation.cpu())
                image_features.append(image_representation.cpu())
        return torch.cat(text_features), torch.cat(image_features)
    
    
if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(folder_path, "train_df_sample.csv"))
    test_df = pd.read_csv(os.path.join(folder_path, "test_df_sample.csv"))
    val_df = pd.read_csv(os.path.join(folder_path, "val_df_sample.csv"))
    train_images = np.load(os.path.join(folder_path, 'train_images.npy'))
    test_images = np.load(os.path.join(folder_path, 'test_images.npy'))
    val_images = np.load(os.path.join(folder_path, 'val_images.npy'))
    model_id = ("zer0int/LongCLIP-GmP-ViT-L-14")
    config = CLIPConfig.from_pretrained(model_id)
    config.text_config.max_position_embeddings = 248
    clip_model = CLIPModel.from_pretrained(model_id, config=config).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=248)
    train_dataset = TextImageDataset(train_df, train_images, clip_processor)
    train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True, collate_fn = collate_fn)
    test_dataset = TextImageDataset(test_df, test_images, clip_processor)
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False, collate_fn = collate_fn)
    valid_dataset = TextImageDataset(val_df, val_images, clip_processor)
    valid_dataloader = DataLoader(valid_dataset, batch_size=200, shuffle=False, collate_fn = collate_fn)
    projection_model = Projection_Model(768)
    model = RepresentationNN(clip_model, projection_model).to(device)
    # Freeze the CLIP/pre-trained model parameters outside the model class
    for param in model.pretrained_model.parameters():
        param.requires_grad = False  # Freeze the parameters
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss = InfoNCE(temperature=0.1, negative_mode="paired")
    eval_loss = InfoNCE(temperature=0.1)
    num_epochs = 20
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )
    train(model, train_dataloader, valid_dataloader, optimizer, loss, num_epochs=num_epochs, lr_scheduler=lr_scheduler, negative = 6)
    model_path = os.path.join(folder_path, "weights", "projection_model.pth")
    torch.save(model.projection_model.state_dict(), model_path)
    combination = {
        "train":{"dataloader":train_dataloader, "len":train_dataset.__len__()},
        "test":{"dataloader":test_dataloader, "len":test_dataset.__len__()},
        "val":{"dataloader":valid_dataloader, "len":valid_dataset.__len__()}
    }
    for key in combination.keys():
        mode = key
        acc1, acc5, mrr = retrieval_eval(combination[mode]["dataloader"], combination[mode]["len"])
        print(f"-----{key}-----")
        print("acc @ 1:", acc1)
        print("acc @ 5:", acc5)
        print("MRR:", mrr)

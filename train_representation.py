import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import IPython.display as display
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from transformers import get_scheduler
from sentence_transformers.util import semantic_search, cos_sim
from info_nce import InfoNCE, info_nce


folder_path = '/content/drive/MyDrive/DeepLearning/Final/SemArt/Target'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Define model class and functions

class TextImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_array: np.array, processor: CLIPProcessor):
        self.len = df.shape[0]
        self.image_array = image_array
        self.processor = processor
        self.text = df['DESCRIPTION'].tolist()

    def __len__(self):
        # Return the number of samples in the dataset
        return self.len

    def __getitem__(self, idx):
        # Get the text and image for the sample at index idx
        text = self.text[idx]
        image = Image.fromarray(self.image_array[idx])  # Convert to PIL Image
        # Preprocess the text and image using CLIP processor
        input_text = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        input_images = self.processor(images=image, return_tensors="pt")
        return input_text, input_images
    
# projection which has been trained in advance
class Projection_Model(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 768):
        super(Projection_Model, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, text_inputs, image_inputs):
        text_embeddings = self.projection(text_inputs)
        image_embeddings = self.projection(image_inputs)
        return text_embeddings, image_embeddings

# ultimate representation model which contains one pretrained representation model and one pretrained projection model
class RepresentationNN(nn.Module):
    def __init__(self, pretrained_model: nn.Module, projection_model: Projection_Model, input_dim: int, hidden_dim: int = 768):
        super(RepresentationNN, self).__init__()
        self.pretrained_model = pretrained_model
        self.projection = projection_model
    def forward(self, text_inputs: dict, image_inputs: dict):
        text_features = self.pretrained_model.get_text_features(**text_inputs)
        image_features = self.pretrained_model.get_image_features(**image_inputs)
        text_embeddings = self.projection(text_features)
        image_embeddings = self.projection(image_features)
        return text_embeddings, image_embeddings

    def get_text_features(self, text_inputs: dict):
        text_features = self.pretrained_model.get_text_features(**text_inputs)
        return self.projection(text_features)

    def get_image_features(self, image_inputs: dict):
        image_features = self.pretrained_model.get_image_features(**image_inputs)
        return self.projection(image_features)

# unused, but functioning equally as InfoNCE Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings to unit vectors (important for cosine similarity)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)

        # Calculate cosine similarity (scaled by temperature)
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        logits_per_text = logits_per_image.T  # same matrix, transposed

        # Labels for contrastive loss (diagonal entries are positive pairs)
        labels = torch.arange(text_embeddings.size(0), device=text_embeddings.device)

        # Cross-entropy loss between text and image embeddings
        loss_image = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_text = torch.nn.functional.cross_entropy(logits_per_text, labels)

        # Combine the losses (image-to-text and text-to-image)
        loss = (loss_image + loss_text) / 2.0
        return loss

# PyTorch's data collator function, ask Zion for more information
def collate_fn(batch):
    text_inputs = [item[0]['input_ids'].squeeze(0) for item in batch]
    attention_masks = [item[0]['attention_mask'].squeeze(0) for item in batch]
    image_inputs = [item[1]['pixel_values'].squeeze(0) for item in batch]
    # Pad text sequences to the same length
    text_inputs_padded = torch.nn.utils.rnn.pad_sequence(text_inputs, batch_first=True)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    # Stack image inputs into a single tensor
    image_inputs_stacked = torch.stack(image_inputs)
    # Return dictionary for text inputs and image inputs
    return {
        "input_text": {"input_ids": text_inputs_padded, "attention_mask": attention_masks_padded},
        "input_images": {"pixel_values": image_inputs_stacked}
    }


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
        evaluate(model, loss_fn, valid_dataloader, negative=negative)

def evaluate(model: nn.Module, loss_fn: nn.Module, dataloader: DataLoader, validation: bool = True, negative: int | None = None):
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
                # labels = torch.ones(text_representation.size(0)).to(device)
                loss = loss_fn(text_representation, image_representation)
                total_loss += loss.item()
            print(f"Evaluation Loss: {total_loss/len(dataloader):.4f}")
        return torch.cat(text_features), torch.cat(image_features)
    
def retrieval_eval(dataloader: DataLoader, size: int) -> tuple:
    text_features, image_features = evaluate(model, eval_loss, dataloader, validation=False)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # Perform semantic search
    hits = semantic_search(text_features, image_features, top_k=5)
    count1 = 0
    count5 = 0
    for i in range(len(hits)):
        hit = [hits[i][j]["corpus_id"] for j in range(len(hits[i]))]
        if hit[0] == i:
            count1 += 1
            count5 += 1
        elif i in hit:
            count5 += 1
        else:
            continue
    return count1/size, count5/size
    
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
    model = RepresentationNN(clip_model, projection_model, 768).to(device)
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
    model_path = os.path.join(folder_path, "projection_model.pth")
    torch.save(model.projection_model.state_dict(), model_path)
    combination = {
        "train":{"dataloader":train_dataloader, "len":train_dataset.__len__()},
        "test":{"dataloader":test_dataloader, "len":test_dataset.__len__()},
        "val":{"dataloader":valid_dataloader, "len":valid_dataset.__len__()}
    }
    for key in combination.keys():
        mode = key
        acc1, acc5 = retrieval_eval(combination[mode]["dataloader"], combination[mode]["len"])
        print(f"-----{key}-----")
        print("acc @ 1:", acc1)
        print("acc @ 5:", acc5)

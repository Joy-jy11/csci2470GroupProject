from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from transformers import CLIPProcessor
from PIL import Image
import pandas as pd
from model import RepresentationNN
# from info_nce import InfoNCE
from train_representation import evaluate
from sentence_transformers.util import semantic_search


class TextImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_array: np.array, processor: CLIPProcessor, column: str = "DESCRIPTION"):
        self.len = df.shape[0]
        self.image_array = image_array
        self.processor = processor
        self.text = df[column].tolist()

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
def retrieval_eval(model: RepresentationNN, dataloader: DataLoader, size: int) -> tuple:
    text_features, image_features = evaluate(model, dataloader, validation=False)
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
    mrr = calculate_mrr(text_features, image_features)
    return count1/size, count5/size, mrr

def calculate_mrr(text_features: torch.Tensor, image_features: torch.Tensor):
    """
    Calculates the Mean Reciprocal Rank (MRR) for a given set of text and image features.

    Args:
        text_features: A tensor of text features.
        image_features: A tensor of image features.
        top_k: The number of top results to consider for MRR calculation.

    Returns:
        The Mean Reciprocal Rank (MRR).
    """
    hits = semantic_search(text_features, image_features, top_k=image_features.shape[0])
    mrr_sum = 0
    for i in range(len(hits)):
        hit = [hits[i][j]["corpus_id"] for j in range(len(hits[i]))]
        mrr_sum += (1 / (1 + hit.index(i)))
    mrr = mrr_sum / len(hits)
    return mrr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import IPython.display as display
# from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from transformers import get_scheduler
from sentence_transformers.util import semantic_search
from info_nce import InfoNCE, info_nce
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from model import RepresentationNN, Projection_Model
from util import TextImageDataset, collate_fn
# from train_representation import evaluate, retrieval_eval
from PIL import Image

folder_path = 'Your data path for sampled data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function that encodes text inputs and search for relevant images
def text_image_search(text: list, image_features: torch.Tensor, model: RepresentationNN, top_k: int = 5) -> list:
    text_input = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
    text_input = {k:v.to(device) for k, v in text_input.items()}
    model.eval()
    with torch.no_grad():
        text_features = model.get_text_features(text_input).cpu()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    hits = semantic_search(text_features, image_features, top_k=top_k)
    index_list = []
    for i in range(len(hits)):
        hit = [hits[i][j]["corpus_id"] for j in range(len(hits[i]))]
        index_list.append(hit)
    return index_list

# function that gets representations for image corpus
def get_image_features(model: RepresentationNN, image_array: np.array, batch_size: int = 100):
    images = [Image.fromarray(image_array[i]) for i in range(image_array.shape[0])]
    image_input = clip_processor(images=images, return_tensors="pt")
    # image_input = {k:v.to(device) for k, v in image_input.items()}
    image_features = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size)):
            image_input_batch = {k: v[i:i+batch_size].to(device) for k, v in image_input.items()}
            image_feature = model.get_image_features(image_input_batch).cpu()
            image_features.append(image_feature)
    image_features = torch.cat(image_features, dim=0)
    return image_features

if __name__ == "__main__":
    """loading models and data"""
    # train_df = pd.read_csv(os.path.join(folder_path, "train_df_sample.csv"))
    train_images = np.load(os.path.join(folder_path, 'train_images.npy'))
    model_id = ("zer0int/LongCLIP-GmP-ViT-L-14")
    config = CLIPConfig.from_pretrained(model_id)
    config.text_config.max_position_embeddings = 248
    clip_model = CLIPModel.from_pretrained(model_id, config=config).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=248)
    # train_dataset = TextImageDataset(train_df, train_images, clip_processor)
    # train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True, collate_fn = collate_fn)
    projection_model_path = os.path.join(folder_path, "weights", "prejection_model.pth")
    projection_model = Projection_Model(768)
    projection_model.load_state_dict(torch.load(projection_model_path, weights_only=True))
    model = RepresentationNN(clip_model, projection_model).to(device)

    """getting features"""
    # get image features, which are later to be searched for
    image_corpus_path = os.path.join(folder_path, "train_image_corpus.pt")
    # image_corpus_path = os.path.join(folder_path, "test_image_corpus.pt")
    if os.path.exists(image_corpus_path):
        image_features = torch.load(image_corpus_path, weights_only = True)
    else:
        image_features = get_image_features(model, train_images)
        torch.save(image_features, image_corpus_path)
    # normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # text input sample; create one using the code above
    text = [
        # This is a shorten description of train_df["DESCRIPTION"][9]. Use ChatGPT do get a shorten descriptoin.
        """
    Create an artwork of train carriage with Dutch Golden Age-inspired scene in the style of Pieter de Hooch, blending still-life qualities with precise perspective, and luminous colors. Capture a serene, frozen moment imbued with timelessness, where the interplay of light and structure evokes a sense of quiet eternity.
        """
    ]
    # index is a list containing lists of indices
    # if you have two queries and are looking for top 5 relevant images, "index" will look like: [ [i1,i2,i3,i4,i5], [i1,i2,i3,i4,i5]]
    # ik is an index for the image corpus
    index = text_image_search(text, image_features, model)
    print("prompt:")
    print(text[0])
    print("ground truth image index:", 9)
    print("retrieved image index:", index[0])
    print("ground truth image:")
    display.display(Image.fromarray(train_images[9]))
    print("retrieved images:")
    fig, axes = plt.subplots(1, len(index[0]), figsize=(15, 5))  # Adjust figsize as needed

    for i, idx in enumerate(index[0]):
        axes[i].imshow(train_images[idx])
        axes[i].axis('off')  # Hide axes
    plt.show()


import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from util import retrieval_eval

folder_path = "<Your data path for sampled data>"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(folder_path: str, df_name: str, image_name: str) -> tuple:
    images = np.load(os.path.join(folder_path, image_name))
    df = pd.read_csv(os.path.join(folder_path, df_name))
    return df, images

def get_text_image_input(clip_processor: CLIPProcessor, df: pd.DataFrame, image_array: np.array, column: str = 'DESCRIPTION') -> dict:
    text = df[column].tolist()
    images = []
    for i in tqdm(range(df.shape[0]), desc = "working on loading images..."):
        images.append(Image.fromarray(image_array[i]))
    input_text = clip_processor(text=text, return_tensors="pt", padding=True, truncation = True).to(device)
    input_images = clip_processor(images=images, return_tensors="pt").to(device)
    return {"input_text":input_text, "input_images": input_images}

def get_text_image_features(clip_model: CLIPModel, inputs: dict, batch_size: int = 100) -> tuple:
    text_features = []
    image_features = []
    clip_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs["input_text"]["input_ids"]), batch_size), desc = "working on getting features..."):
            s, e = i, i + batch_size
            # text_feature = blip_model.get_text_features(**inputs["input_text"][s:e])
            # image_feature = blip_model.get_image_features(**inputs["input_images"][s:e])
            text_feature = clip_model.get_text_features(**{k: v[s:e].to(device) for k, v in inputs["input_text"].items()})
            image_feature = clip_model.get_image_features(**{k: v[s:e].to(device) for k, v in inputs["input_images"].items()})
            text_features.append(text_feature)
            image_features.append(image_feature)
    text_features = torch.cat(text_features, dim=0)
    image_features = torch.cat(image_features, dim=0)
    return text_features, image_features

if __name__ == "__main__":
    train_df, train_images = load_data(folder_path, "train_df_sample.csv", "train_images.npy")
    val_df, val_images = load_data(folder_path, "val_df_sample.csv", "val_images.npy")
    test_df, test_images = load_data(folder_path, "test_df_sample.csv", "test_images.npy")
    model_id = ("zer0int/LongCLIP-GmP-ViT-L-14")
    config = CLIPConfig.from_pretrained(model_id)
    config.text_config.max_position_embeddings = 248
    clip_model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, config=config).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=248)
    for task_type in ["train", "val", "test"]:
        df, images =load_data(folder_path, f"{task_type}_df_sample.csv", f"{task_type}_images.npy")
        inputs = get_text_image_input(clip_processor, df, images)
        text_features, image_features = get_text_image_features(clip_model, inputs)
        acc1, acc5 = retrieval_eval(text_features, image_features)
        print(f"-----{task_type}-----")
        print("acc @ 1:", acc1)
        print("acc @ 5:", acc5)
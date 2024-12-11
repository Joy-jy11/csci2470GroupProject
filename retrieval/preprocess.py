import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import re

folder_path = '<Your data path for original dataset>'
target_path = "Your data path for sampled data"


def read_to_csv(folder_path: str, target_path: str, file_name: str, delimiter: str = '\t', encoding='latin1', n = 2000):
    df = pd.read_csv(os.path.join(folder_path, file_name), delimiter=delimiter, encoding=encoding)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df_sample = df[:n]
    sample_type = re.search(r"_(.*?)\.", file_name).group(1)
    df_sample.to_csv(os.path.join(target_path, f"{sample_type}_df_sample.csv"), drop = True)
    read_and_save_images(df_sample, target_path, f"{sample_type}_images.npy")


def read_and_save_images(df: pd.DataFrame, folder_path: str, filename: str):
    image_arrays = []
    image_folder = os.path.join(folder_path, "Images")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_path = os.path.join(image_folder, row['IMAGE_FILE'])
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_arrays.append(image_array)
    np.save(os.path.join(target_path, filename), np.stack(image_arrays))

if __name__ == "__main__":
    n = 200
    read_to_csv(folder_path, target_path, "semart_train.csv")
    read_to_csv(folder_path, target_path, "semart_test.csv", n = n)
    read_to_csv(folder_path, target_path, "semart_val.csv", n = n)


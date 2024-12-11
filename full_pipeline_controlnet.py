import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
from PIL import Image
from controlnet_aux import CannyDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
from diffusers.utils import load_image, make_image_grid
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from retrieval.model import RepresentationNN, Projection_Model
from retrieval.util import TextImageDataset, collate_fn
from retrieval.inference import get_image_features, text_image_search

folder_path = 'Your data path for sampled'
projection_model_path = os.path.join("./retrieval/weights", "projection_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = ("zer0int/LongCLIP-GmP-ViT-L-14")
diff_model_id = "runwayml/stable-diffusion-v1-5"
K = 5

if __name__ == "__main__":
    train_images = np.load(os.path.join(folder_path, 'train_images.npy'))
    config = CLIPConfig.from_pretrained(model_id)
    config.text_config.max_position_embeddings = 248
    clip_model = CLIPModel.from_pretrained(model_id, config=config).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=248)
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
    index = text_image_search(text, image_features, model, top_k=K)
    """check retrived images"""
    print("prompt:")
    print(text[0])
    print("retrieved image index:", index[0])
    print("retrieved images:")
    fig, axes = plt.subplots(1, len(index[0]), figsize=(15, 5))  # Adjust figsize as needed
    for i, idx in enumerate(index[0]):
        axes[i].imshow(train_images[idx])
        axes[i].axis('off')  # Hide axes
    plt.show()
    """loading diffusion-related models"""
    sd_pipe = StableDiffusionPipeline.from_pretrained(diff_model_id, torch_dtype=torch.float16).to(device)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16,
        varient="fp16"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        diff_model_id,
        controlnet=controlnet,
        image_encoder=image_encoder,
        torch_dtype=torch.float16
    ).to(device)
    pipe.load_ip_adapter("h94/IP-Adapter",
                        subfolder="models",
                        weight_name=["ip-adapter_sd15.bin" for _ in range(K)]
    )
    pipe.enable_model_cpu_offload()
    pipe.set_ip_adapter_scale([0.2, 0.2, 0.2, 0.2, 0.2])
    prompt = text[0]
    idx = index[0]
    image = sd_pipe(prompt, num_inference_steps=20, negative_prompt = "no sensitive and naked/nude content; No religiously-looking content").images[0]
    # loading images for ip adapter
    ip_adap_img = [Image.fromarray(train_images[idx[i]]) for i in range(len(idx))]
    # detect edges
    canny = CannyDetector()
    canny_img = canny(image, detect_resolution=512, image_resolution=768)
    images = pipe(prompt = prompt,
              negative_prompt = "no low quality; don't change the skin color of the main figure",
              height = 768,
              width = 768,
              ip_adapter_image = ip_adap_img,
              image = canny_img,
              guidance_scale = 6,
              controlnet_conditioning_scale = 0.7,
              num_inference_steps = 20,
              num_images_per_prompt = 3).images
    images = [image] + images
    make_image_grid(images, cols=4, rows=1)
    # Create the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    # Iterate through the images and save them
    for i, image in enumerate(images):
        image_path = os.path.join(folder_path, f"generated_sdcn_image_{i}.png")  # Customize the filename as needed
        image.save(image_path)






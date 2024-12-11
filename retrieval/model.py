import torch.nn as nn
import torch
from transformers import CLIPModel

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
    def __init__(self, pretrained_model: CLIPModel, projection_model: Projection_Model):
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
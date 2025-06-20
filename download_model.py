from open_clip import create_model_and_transforms, get_tokenizer
import torch
import os

# Create directory if it doesn't exist
os.makedirs('./model', exist_ok=True)

# Load model and download weights (this caches the model)
model, _, _ = create_model_and_transforms(
    model_name="ViT-L-14-336",
    pretrained="openai"
)

# Save the state_dict to ./model
torch.save(model.state_dict(), './model/openclip-vit-l-14-336px.pth')
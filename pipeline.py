import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file
from collections import OrderedDict
from src.models.mamba_models import AudioMamba

# Path to dataset
DATA_PATH = "data/dev_spectrograms/"

# Define preprocessing for spectrogram normalization (if needed)
transform = transforms.Compose([
    transforms.Resize((128, 1024)),  # Match model input size
    transforms.ToTensor(),  # Normalize [0,1]
])

# Load safetensors checkpoint
checkpoint_path = "model.safetensors"  # Update with actual path
state_dict = load_file(checkpoint_path)

# Remove "model." prefix from keys
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("model.", "")
    new_state_dict[new_key] = v

# Initialize Audio-Mamba Model
aum_model = AudioMamba()

# Remove classification head from checkpoint
del new_state_dict["head.weight"]
del new_state_dict["head.bias"]

# Load the modified state_dict
aum_model.load_state_dict(new_state_dict, strict=False)
aum_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aum_model = aum_model.to(device)

def load_spectrograms_from_folder(folder_path):
    """Loads all spectrogram images from a folder and returns a list of tensors."""
    spectrograms = []
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):  # Adjust based on file format
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("L")  # Convert to grayscale
            
            spectrogram_tensor = transform(image)  # Apply preprocessing
            spectrogram_tensor = spectrogram_tensor.unsqueeze(0)  # Add batch dim
            spectrograms.append(spectrogram_tensor)

    return spectrograms

# Load train and test spectrograms
train_folder = "./data/dev_spectrograms/bearing/train/"
test_folder = "./data/dev_spectrograms/bearing/test/"

train_spectrograms = load_spectrograms_from_folder(train_folder)
test_spectrograms = load_spectrograms_from_folder(test_folder)

print(f"Loaded {len(train_spectrograms)} training spectrograms and {len(test_spectrograms)} test spectrograms.")
def extract_features(aum_model, spectrograms):
    """Extract embeddings from a list of spectrograms using Audio-Mamba."""
    embeddings = []

    for spectrogram in spectrograms:
        spectrogram = spectrogram.to(device)  # Move to GPU if available
        spectrogram = spectrogram.squeeze(3)  # Fix shape issue (B, C, H, W) → (B, C, H)
        
        with torch.no_grad():
            embedding = aum_model(spectrogram).cpu().numpy()  # Extract embeddings
        embeddings.append(embedding)

    return np.array(embeddings).squeeze()  # Shape: (num_samples, 527)

# Extract embeddings
train_embeddings = extract_features(aum_model, train_spectrograms)
test_embeddings = extract_features(aum_model, test_spectrograms)

print("Train Embeddings Shape:", train_embeddings.shape)  # Should be (num_samples, 527)
print("Test Embeddings Shape:", test_embeddings.shape)



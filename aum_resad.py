import torch
import torch
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file
from collections import OrderedDict
from src.models.mamba_models import AudioMamba

image_path = "./data/dev_spectrograms/bearing/train/section_00_source_train_normal_0001_pro_A_vel_4_loc_A.jpg"
image = Image.open(image_path).convert("L")
transform = transforms.Compose([
    transforms.Resize((128, 1024)),  # Resize to match expected model input size
    transforms.ToTensor(),  # Convert to tensor (automatically normalizes to [0,1])
])
spectrogram_tensor = transform(image)
spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
# Load safetensors checkpoint
checkpoint_path = "model.safetensors"
state_dict = load_file(checkpoint_path)

# Remove "model." prefix from keys
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("model.", "")
    new_state_dict[new_key] = v

# Initialize your model
aum_model = AudioMamba()
# Remove classification head from checkpoint
del new_state_dict["head.weight"]
del new_state_dict["head.bias"]

aum_model.eval()  # Set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spectrogram_tensor = spectrogram_tensor.to(device)
aum_model = aum_model.to(device)

# Load the modified state_dict
aum_model.load_state_dict(new_state_dict, strict=False)
print("Input Tensor Shape:", spectrogram_tensor.shape)
spectrogram_tensor = spectrogram_tensor.squeeze(3)
print("Fixed Tensor Shape:", spectrogram_tensor.shape)
embeddings = aum_model(spectrogram_tensor)

print("Audio-Mamba Output Shape:", embeddings.shape)

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dcase import DCASE2024  # Assuming `dcase.py` is inside `datasets/`

# Set dataset root and machine types
dataset_root = "data/dev_spectrograms"  # Update this path based on your dataset structure
machine_types = ["fan", "gearbox", "bearing"]  # Example subset, include all types as needed

# Create dataset instance (for training)
train_dataset = DCASE2024(root=dataset_root, machine_types=machine_types, train=True)

# Create dataset instance (for testing)
test_dataset = DCASE2024(root=dataset_root, machine_types=machine_types, train=False)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Print dataset info
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Number of machine types: {len(machine_types)}")

def show_spectrogram(image, label, class_name):
    """Displays a spectrogram image with class label."""
    image = image.squeeze(0).cpu().numpy()  # Convert tensor to numpy (Grayscale)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="magma")  # Use 'magma' colormap for better contrast
    plt.title(f"Class: {class_name} (Label: {label})")
    plt.axis("off")
    plt.show()

# Get a single batch
data_iter = iter(train_loader)
images, labels, class_names = next(data_iter)

# Show first 5 spectrograms
for i in range(5):
    show_spectrogram(images[i], labels[i].item(), class_names[i])

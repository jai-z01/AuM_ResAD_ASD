import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def wav_to_mel_spectrogram(wav_path, output_path, sr=16000, n_mels=128, fmax=8000):
    # Load audio
    y, sr = librosa.load(wav_path, sr=sr)
    
    # Compute log mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot and save spectrogram as .jpg
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', fmax=fmax)
    plt.axis('off')  # Remove axes for cleaner image
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_directory(input_dir, output_dir, sr=16000, n_mels=128, fmax=8000):
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                relative_path = os.path.relpath(wav_path, input_dir)  # Keep folder structure
                output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.jpg')
                
                # Convert and save spectrogram
                try:
                    wav_to_mel_spectrogram(wav_path, output_path, sr=sr, n_mels=n_mels, fmax=fmax)
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")

if __name__ == "__main__":
    # Input and output directories
    input_directory = "./data/eval"  # Replace with the path to your .wav files
    output_directory = "./data/eval_spectrograms"  # Replace with the path for spectrograms
    
    # Process the .wav files
    process_directory(input_directory, output_directory)

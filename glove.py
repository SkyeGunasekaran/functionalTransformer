import requests
import zipfile
import io
import torch
import numpy as np
from tqdm import tqdm
import os

# --- Configuration ---
GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.50d.txt"
OUTPUT_FILE = "glove_data.pt"

def download_and_extract(url, target_file):
    """Downloads and extracts the GloVe file."""
    if not os.path.exists(target_file):
        print(f"Downloading {url}...")
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            print(f"Failed to download: Status {r.status_code}")
            return False
        
        print("Download complete. Extracting...")
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            if target_file in z.namelist():
                z.extract(target_file)
                print(f"Extracted {target_file}")
            else:
                print(f"Error: {target_file} not found in zip archive.")
                return False
    else:
        print(f"{target_file} already exists. Skipping download.")
    return True

def process_glove_file(glove_file, output_file):
    """Reads the .txt file and saves to a .pt file."""
    print(f"Processing {glove_file}...")
    
    words = []
    word_to_idx = {}
    embeddings = []
    
    with open(glove_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=400000)): # 400k lines in file                
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]])
            
            words.append(word)
            word_to_idx[word] = i
            embeddings.append(vector)
            
    # Convert to Tensors
    glove_vectors = torch.tensor(np.array(embeddings), dtype=torch.float32)
    
    # Save the processed data
    data_to_save = {
        'words': words,
        'word_to_idx': word_to_idx,
        'vectors': glove_vectors
    }
    torch.save(data_to_save, output_file)
    print(f"Successfully processed {len(words)} words.")
    print(f"Saved data to {output_file}")

if __name__ == "__main__":
    if download_and_extract(GLOVE_URL, GLOVE_FILE):
        process_glove_file(GLOVE_FILE, OUTPUT_FILE)

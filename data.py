import torch
from torch.utils.data import DataLoader

# Standard Python libraries for data loading
import os
import requests  # To download the data
import zipfile   # To unzip
import sys       # For printing progress
from collections import Counter

class WikiText103Dataset(Dataset):
    """A custom torch.utils.data.Dataset for Wikitext-103."""
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # Return the number of full blocks, minus one for the shift
        return (self.data.numel() - 1) // self.block_size

    def __getitem__(self, idx):
        # Get a chunk of data
        start = idx * self.block_size
        end = start + self.block_size
        
        # x is the input sequence
        x = self.data[start:end]
        # y is the target sequence, shifted by one
        y = self.data[start+1:end+1]
        return x, y
    
class Wikitext103Loader:
    """
    Downloads, processes, and loads Wikitext-103 without torchtext.
    """
    def __init__(self, data_dir, block_size, batch_size, min_freq=100):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
        self.zip_path = os.path.join(data_dir, "wikitext-103-v1.zip")
        self.extracted_path = os.path.join(data_dir, "wikitext-103")
        self.train_path = os.path.join(self.extracted_path, "wiki.train.tokens")
        self.val_path = os.path.join(self.extracted_path, "wiki.valid.tokens")
        
        # Vocab attributes
        self.stoi = {} # string to int
        self.itos = {} # int to string
        self.vocab_size = 0
        
        # Data attributes
        self.train_data = None
        self.val_data = None

    def _download_progress(self, stream, file_path, total_size):
        """Helper to show download progress."""
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in stream.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r  Downloading... {progress:.2f}%")
                    sys.stdout.flush()
        print() # Newline after download

    def _download_and_extract(self):
        """Downloads and extracts the dataset."""
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.train_path):
            if not os.path.exists(self.zip_path):
                print(f"Wikitext-103 not found. Downloading to {self.zip_path}...")
                try:
                    with requests.get(self.url, stream=True) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get('content-length', 0))
                        if total_size == 0:
                            raise Exception("Could not get file size.")
                        self._download_progress(r, self.zip_path, total_size)
                    print("Download complete.")
                except Exception as e:
                    print(f"Failed to download: {e}")
                    if os.path.exists(self.zip_path): os.remove(self.zip_path)
                    return
            
            print(f"Extracting {self.zip_path}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("Extraction complete.")
        else:
            print("Wikitext-103 files found.")

    def _build_vocab(self):
        """Builds a vocabulary from the training data."""
        print(f"Building vocabulary (min_freq={self.min_freq})...")
        word_counts = Counter()
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                word_counts.update(line.strip().split())
        
        # Create vocab
        specials = ['<unk>', '<pad>']
        self.stoi = {token: i for i, token in enumerate(specials)}
        self.itos = {i: token for i, token in enumerate(specials)}
        
        # Add words that meet min_freq
        for word, count in word_counts.items():
            if count >= self.min_freq:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word
        
        self.vocab_size = len(self.stoi)
        print(f"Vocabulary built. Size: {self.vocab_size}")

    def _tokenize(self, file_path):
        """Numericalizes a text file into a single torch.LongTensor."""
        print(f"Tokenizing {os.path.basename(file_path)}...")
        token_ids = []
        unk_idx = self.stoi['<unk>']
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                token_ids.extend([self.stoi.get(word, unk_idx) for word in words])
        return torch.LongTensor(token_ids)

    def load(self):
        """Main method to load and process data."""
        self._download_and_extract()
        self._build_vocab()
        
        self.train_data = self._tokenize(self.train_path)
        self.val_data = self._tokenize(self.val_path)
        
        train_dataset = WikiText103Dataset(self.train_data, self.block_size)
        val_dataset = WikiText103Dataset(self.val_data, self.block_size)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=True,
            num_workers=4
        )
        
        print("DataLoaders created.")
        return train_loader, val_loader, self.vocab_size

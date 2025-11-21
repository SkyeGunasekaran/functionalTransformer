import os
import torch
import numpy as np
from datasets import load_dataset 
import tiktoken 
from torch.utils.data import Dataset, DataLoader

class WikitextDataset(Dataset):
    def __init__(self, data_dir, split, block_size):
        self.block_size = block_size
        filename = os.path.join(data_dir, f"{split}.bin")
        
        # 1. Load or Create the binary file (Memory Mapped)
        if not os.path.exists(filename):
            self._create_bin_file(filename, split)
        
        # Load as a memory map
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        
        # CRITICAL FIX 1: Calculate length based on BLOCKS, not tokens
        # We divide total tokens by block_size to get number of full sequences
        self.num_samples = (len(self.data) - 1) // self.block_size
        
        print(f"Loaded {split} data from {filename}")
        print(f"Tokens: {len(self.data)/1e6:.2f}M | Sequences: {self.num_samples}")

    def _create_bin_file(self, filename, split):
        print(f"--- Pre-processing {split} split (One time only) ---")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        tokenizer = tiktoken.get_encoding("gpt2")
        
        tokenized = []
        for item in dataset:
            if len(item['text']) > 0:
                tokenized.extend(tokenizer.encode(item['text']))
                tokenized.append(tokenizer.eot_token)
        
        arr = np.array(tokenized, dtype=np.uint16)
        with open(filename, 'wb') as f:
            f.write(arr.tobytes())
        print(f"Saved binary to {filename}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # CRITICAL FIX 2: Stride by block_size
        # Instead of idx, we start at idx * block_size
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1
        
        # Slice
        chunk = torch.from_numpy((self.data[start_idx : end_idx]).astype(np.int64))
        
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloaders(batch_size, block_size, num_workers=4):
    data_dir = "data/wikitext_cache"
    
    # Instantiate tokenizer once just to get the real vocab size
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    train_ds = WikitextDataset(data_dir, "train", block_size)
    val_ds = WikitextDataset(data_dir, "validation", block_size)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,
        drop_last=True 
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,
        drop_last=True
    )
    
    return train_loader, val_loader, vocab_size

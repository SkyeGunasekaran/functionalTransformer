import torch
from tqdm import tqdm
import tiktoken
from datasets import load_dataset

# ==============================================================================
# 2. ROBUST DATA LOADER (Wikitext-103 + TikToken)
# ==============================================================================

class WikitextLoader:
    def __init__(self, block_size, batch_size):
        self.block_size = block_size
        self.batch_size = batch_size
        self.tokenizer = tiktoken.get_encoding("gpt2") # Standard 50k Vocab
        print("Loading Wikitext-103 from HuggingFace...")
        # Using 'wikitext-2-v1' for faster debugging, switch to 'wikitext-103-v1' for full run
        self.dataset = load_dataset("wikitext", "wikitext-103-v1") 
        
    def prepare_data(self, split):
        data = self.dataset[split]['text']
        # Flatten and encode
        print(f"Tokenizing {split} split...")
        tokenized = []
        for text in tqdm(data, desc=f"Encoding {split}"):
            if len(text) > 0:
                tokenized.extend(self.tokenizer.encode(text))
                tokenized.append(self.tokenizer.eot_token) # End of article
        
        return torch.tensor(tokenized, dtype=torch.long)

    def get_loader(self, split):
        data = self.prepare_data(split)
        n_batches = len(data) // (self.batch_size * self.block_size)
        # Trim to fit perfectly
        data = data[:n_batches * self.batch_size * self.block_size]
        x = torch.stack([data[i:i+self.block_size] for i in range(0, len(data), self.block_size)])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in range(0, len(data), self.block_size)])
        
        # Reshape into batches
        # This simple loader yields pre-batched tensors for speed
        x = x.view(-1, self.batch_size, self.block_size)
        y = y.view(-1, self.batch_size, self.block_size)
        
        return x, y, self.tokenizer.n_vocab

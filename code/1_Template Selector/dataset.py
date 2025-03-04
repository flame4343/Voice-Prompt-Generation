import torch
from torch.utils.data import Dataset
import random


class TemplateDataset(Dataset):
    """
    Custom Dataset for loading source, target, and template text data.
    """

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        source = entry['source']
        target = entry['target']
        templates = entry['template']

        # Tokenizing input text
        source_tokens = self.tokenizer(source, padding='max_length', truncation=True, return_tensors='pt',
                                       max_length=self.max_length)
        target_tokens = self.tokenizer(target, padding='max_length', truncation=True, return_tensors='pt',
                                       max_length=self.max_length)

        negative_template = random.choice(templates)
        negative_tokens = self.tokenizer(negative_template, padding='max_length', truncation=True, return_tensors='pt',
                                         max_length=self.max_length)

        return source_tokens['input_ids'].squeeze(0), target_tokens['input_ids'].squeeze(0), negative_tokens[
            'input_ids'].squeeze(0), templates


def collate_fn(batch):
    """
    Collate function to handle batching of dataset.
    """
    source_input_ids = torch.stack([item[0] for item in batch], dim=0)
    target_input_ids = torch.stack([item[1] for item in batch], dim=0)
    negative_input_ids = torch.stack([item[2] for item in batch], dim=0)
    templates = [item[3] for item in batch]

    return source_input_ids, target_input_ids, negative_input_ids, templates
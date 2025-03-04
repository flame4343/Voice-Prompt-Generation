import json
import re
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, max_elements=20):
        """
        Custom dataset for handling tokenization and label alignment.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_elements = max_elements

    def parse_source(self, source_str):
        """
        Parses source string into a dictionary.
        """
        pattern = r'([^:：]+):\s*([^:：]+)(?=\s[^:：]+:|$)'
        matches = re.findall(pattern, source_str)
        source_dict = {key.strip(): value.strip().rstrip(',') for key, value in matches}
        return source_dict

    def combine_source_elements(self, source_dict, separator=" [SEP] "):
        """
        Combines source elements into a single string and records positions.
        """
        combined = ""
        positions = []
        current_pos = 0
        for key, value in source_dict.items():
            element = f"{key}: {value}"
            if combined:
                combined += separator
                current_pos += len(separator)
            combined += element
            start, end = current_pos, current_pos + len(element)
            positions.append((element, start, end))
            current_pos = end
        return combined, positions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a data sample by index.
        """
        item = self.data[idx]
        source_str, target = item['source'], item['target']
        binary_label = torch.tensor(json.loads(item['binary_label']), dtype=torch.float)

        source_dict = self.parse_source(source_str)
        combined_source, positions = self.combine_source_elements(source_dict)

        inputs = self.tokenizer(
            combined_source, truncation=True, max_length=self.max_length, return_offsets_mapping=True
        )
        input_ids, attention_mask, offset_mapping = inputs['input_ids'], inputs['attention_mask'], inputs['offset_mapping']

        target_inputs = self.tokenizer(target, truncation=True, max_length=self.max_length)
        target_ids, target_attention_mask = target_inputs['input_ids'], target_inputs['attention_mask']

        if len(binary_label) > self.max_elements:
            binary_label = binary_label[:self.max_elements]
            positions = positions[:self.max_elements]
        else:
            padding_length = self.max_elements - len(binary_label)
            binary_label = torch.cat([binary_label, torch.zeros(padding_length)], dim=0)
            positions += [("PAD", 0, 0)] * padding_length

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'offset_mapping': offset_mapping,
            'target_ids': target_ids,
            'target_attention_mask': target_attention_mask,
            'binary_label': binary_label,
            'positions': positions
        }

def load_json_data(file_path):
    """
    Loads data from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    """
    input_ids = [torch.tensor(sample['input_ids']) for sample in batch]
    attention_mask = [torch.tensor(sample['attention_mask']) for sample in batch]
    offset_mapping = [torch.tensor(sample['offset_mapping']) for sample in batch]
    binary_label = torch.stack([sample['binary_label'] for sample in batch], dim=0)
    positions = [sample['positions'] for sample in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    offset_mapping = torch.nn.utils.rnn.pad_sequence(offset_mapping, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'offset_mapping': offset_mapping,
        'binary_label': binary_label,
        'positions': positions
    }

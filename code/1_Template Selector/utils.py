import torch
import json

def hinge_loss(scores, margin=1.0):
    """
    Hinge loss function for ranking.
    """
    loss = torch.nn.functional.relu(margin - (torch.unsqueeze(scores[:, 0], -1) - scores[:, 1:]))
    return torch.mean(loss)


def load_data(training_data_path, percentage=0.1):
    """
    Load training data from JSON file.
    """
    with open(training_data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data[:int(len(data) * percentage)]
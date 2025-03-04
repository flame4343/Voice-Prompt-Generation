import torch
import torch.nn as nn
from transformers import BertModel

class TemplateSelector(nn.Module):
    """
    Model for template selection using BERT embeddings.
    """
    def __init__(self, pretrained_model_name='../../data/bert-base-chinese', embed_dim=768):
        super(TemplateSelector, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.embed_dim = embed_dim
        self.final_linear = nn.Linear(self.embed_dim * 2, 1)
        nn.init.xavier_uniform_(self.final_linear.weight)

    def forward(self, source, template):
        source_output = self.bert(source)[0][:, 0, :]
        template_output = self.bert(template)[0][:, 0, :]
        input_representation = torch.cat([source_output, template_output], dim=1)
        score = self.final_linear(input_representation)
        return score
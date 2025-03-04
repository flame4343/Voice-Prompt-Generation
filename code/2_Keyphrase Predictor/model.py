import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, dropout=0.2):
        """
        BiLSTM-based classifier for binary classification.
        """
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, attention_mask, offset_mapping, positions):
        """
        Forward pass through the model.
        """
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)

        batch_size = input_ids.size(0)
        element_reps = []
        for i in range(batch_size):
            token_offsets = offset_mapping[i].tolist()
            sample_reps = []
            for (elem_text, start, end) in positions[i]:
                if start == 0 and end == 0:
                    sample_reps.append(torch.zeros(lstm_out.size(2), device=lstm_out.device))
                else:
                    token_indices = [j for j, (s, e) in enumerate(token_offsets) if s >= start and e <= end]
                    if token_indices:
                        token_hidden = lstm_out[i, token_indices, :]
                        avg_hidden = token_hidden.mean(dim=0)
                    else:
                        avg_hidden = torch.zeros(lstm_out.size(2), device=lstm_out.device)
                    sample_reps.append(avg_hidden)
            sample_reps = torch.stack(sample_reps, dim=0)
            element_reps.append(sample_reps)
        element_reps = torch.stack(element_reps, dim=0)
        logits = self.fc(element_reps).squeeze(-1)
        return logits

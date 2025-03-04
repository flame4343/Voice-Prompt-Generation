import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from dataset import CustomDataset, load_json_data, collate_fn
from model import BiLSTMClassifier

# Hyperparameters
num_epochs = 10
batch_size = 12
learning_rate = 1e-5

# Load tokenizer and data
tokenizer = BertTokenizerFast.from_pretrained('../../data/bert-base-chinese')
data = load_json_data("../../data/3_key_train.json")

# Split dataset
random.shuffle(data)
n = len(data)
train_data, val_data, test_data = data[:int(0.8*n)], data[int(0.8*n):int(0.9*n)], data[int(0.9*n):]

# Prepare datasets and dataloaders
train_dataset = CustomDataset(train_data, tokenizer, max_length=128, max_elements=12)
val_dataset = CustomDataset(val_data, tokenizer, max_length=128, max_elements=12)
test_dataset = CustomDataset(test_data, tokenizer, max_length=128, max_elements=12)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier(tokenizer.vocab_size, embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.2).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        offset_mapping = batch['offset_mapping'].to(device)
        binary_label = batch['binary_label'].to(device).float()
        positions = batch['positions']

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, offset_mapping, positions)
        loss = criterion(logits, binary_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "bilstm_classifier.pth")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TemplateDataset, collate_fn
from model import TemplateSelector
from utils import hinge_loss, load_data
from transformers import BertTokenizer

def train_epoch(model, optimizer, data_loader, margin=1.0):
    model.train()
    total_loss = 0
    for source_input_ids, target_input_ids, negative_input_ids, _ in tqdm(data_loader, desc="Training", ncols=100, leave=False):
        source_input_ids, target_input_ids, negative_input_ids = source_input_ids.cuda(), target_input_ids.cuda(), negative_input_ids.cuda()
        positive_score = model(source_input_ids, target_input_ids)
        negative_score = model(source_input_ids, negative_input_ids)
        loss = hinge_loss(torch.cat([positive_score, negative_score], dim=1), margin)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

if __name__ == "__main__":
    training_data_path = '../../data/4_template_train.json'
    tokenizer = BertTokenizer.from_pretrained('../../data/bert-base-chinese')
    model = TemplateSelector().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    training_data = load_data(training_data_path, percentage=1)
    dataset = TemplateDataset(training_data, tokenizer)
    data_loader = DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)
    for epoch in range(10):
        print(f"Starting Epoch {epoch + 1}")
        avg_loss = train_epoch(model, optimizer, data_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "template.pth")
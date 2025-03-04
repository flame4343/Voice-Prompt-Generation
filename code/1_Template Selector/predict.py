import torch
import json
from tqdm import tqdm
from transformers import BertTokenizer
from model import TemplateSelector
from utils import load_data

if __name__ == "__main__":
    training_data_path = '../../data/4_template_train.json'
    tokenizer = BertTokenizer.from_pretrained('../../data/bert-base-chinese')
    model = TemplateSelector().cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("template.pth", map_location=device))
    training_data = load_data(training_data_path, percentage=1)
    results = []
    model.eval()
    for entry in tqdm(training_data, desc="Predicting", ncols=100):
        source = entry['source']
        target = entry['target']
        templates = entry['template']
        source_tokens = tokenizer(source, padding=True, truncation=True, return_tensors='pt', max_length=128)
        source_input_ids = source_tokens['input_ids'].cuda()
        template_scores = [(template, model(source_input_ids, tokenizer(template, return_tensors='pt')['input_ids'].cuda()).item()) for template in templates]
        top3_templates = sorted(template_scores, key=lambda x: x[1], reverse=True)[:3]
        output_str = "#".join([item[0] for item in top3_templates])
        results.append({
            'source': source,
            'target': target,  # target
            'template': templates,  # template
            # 'top3templates': [{'template': tpl, 'score': score} for tpl, score in top3_templates],
            'templates_selector': output_str
        })
    with open('../data/4_template_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
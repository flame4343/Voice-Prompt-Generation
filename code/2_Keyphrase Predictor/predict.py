import json
import torch
from transformers import BertTokenizerFast
from model import BiLSTMClassifier
from utils import parse_source, combine_source_elements
from tqdm import tqdm


def predict(source_str, model, tokenizer, device, max_length=128, max_elements=12):
    """
    Perform inference on a single source string.
    """
    source_dict = parse_source(source_str)
    combined_source, positions = combine_source_elements(source_dict)

    inputs = tokenizer(
        combined_source, truncation=True, max_length=max_length, return_offsets_mapping=True, return_tensors="pt"
    )
    input_ids, attention_mask, offset_mapping = inputs['input_ids'], inputs['attention_mask'], inputs['offset_mapping']

    if len(positions) > max_elements:
        positions = positions[:max_elements]
    elif len(positions) < max_elements:
        positions += [("PAD", 0, 0)] * (max_elements - len(positions))

    model.eval()
    with torch.no_grad():
        logits = model(input_ids.to(device), attention_mask.to(device), offset_mapping.to(device), [positions])
        preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(0).cpu().tolist()

    return {elem_text: pred for (elem_text, _, _), pred in zip(positions, preds)}


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained('../../data/bert-base-chinese')
    model = BiLSTMClassifier(tokenizer.vocab_size, embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load("bilstm_classifier.pth", map_location=device))
    model.to(device)

    test_file = "../../data/3_key_train.json"
    output_file = "../../data/3_key_result.json"

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    results_list = []
    for sample in tqdm(test_data, desc="Processing Samples"):
        source_str = sample['source']
        pred_result = predict(source_str, model, tokenizer, device)
        result = " ".join(key for key, value in pred_result.items() if value == 1 and key != "PAD")
        results_list.append({
            "source": source_str,
            "keyphrase_predictor": result
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)

    print(f"Inference completed, results saved to {output_file}")
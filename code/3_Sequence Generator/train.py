import os
import torch.cuda
from datetime import datetime
from torch.optim import AdamW
from config import set_args
from torch.nn import CrossEntropyLoss
from tokenizer import T5PegasusTokenizer
from torch.utils.data import DataLoader
from data_helper import MT5Dataset, load_data, collate_fn
from transformers import get_linear_schedule_with_warmup
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def split_data(data, train_size=0.8, val_size=0.1, random_state=42):
    """
    Splits the dataset into training, validation, and test sets.
    :param train_size: Proportion of data allocated to training.
    :param val_size: Proportion allocated to validation.
    :param random_state: Random seed for reproducibility.
    """
    test_size = 1 - train_size - val_size
    train_data, temp_data = train_test_split(data, test_size=(val_size + test_size), random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)),
                                           random_state=random_state)
    return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)


def calc_loss(logits, decoder_input_ids, decoder_attention_mask, loss_fct):
    """
    Computes the loss for the given logits and decoder inputs.
    """
    decoder_mask = decoder_attention_mask[:, 1:].reshape(-1).bool()
    logits = logits[:, :-1].reshape((-1, logits.size(-1)))[decoder_mask]
    labels = decoder_input_ids[:, 1:].reshape(-1)[decoder_mask]
    return loss_fct(logits, labels)


def calculate_acc(logit, labels, ignore_index):
    """
    Computes the accuracy of model predictions.
    """
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)
    _, logit = logit.max(dim=-1)
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct / n_word if n_word > 0 else 0


def train_epoch(model, dataloader, optimizer, scheduler, loss_fct, tokenizer, args):
    """
    Trains the model for one epoch.
    """
    model.train()
    epoch_loss, epoch_acc = 0, 0
    progress_bar = tqdm(dataloader, desc="Training", position=0, leave=True)

    for step, batch in enumerate(progress_bar):
        if torch.cuda.is_available():
            batch = tuple(t.cuda() for t in batch)

        input_ids, input_mask, decoder_input_ids, decoder_attention_mask = batch
        output = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids,
                       decoder_attention_mask=decoder_attention_mask)
        logits = output.logits
        loss = calc_loss(logits, decoder_input_ids, decoder_attention_mask, loss_fct)
        accuracy = calculate_acc(logits, decoder_input_ids, ignore_index=tokenizer.pad_token_id)

        loss.backward()

        if (step + 1) % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{accuracy:.4f}", "LR": f"{lr:.6f}"})
        epoch_loss += loss.item()
        epoch_acc += accuracy

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def evaluate(model, dataloader, loss_fct, tokenizer, desc="Validation"):
    """
    Evaluates the model on validation or test data.
    """
    model.eval()
    eval_loss, eval_acc = 0, 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc, position=0, leave=True)
        for batch in progress_bar:
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            input_ids, input_mask, decoder_input_ids, decoder_attention_mask = batch
            output = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask)
            logits = output.logits
            loss = calc_loss(logits, decoder_input_ids, decoder_attention_mask, loss_fct)
            accuracy = calculate_acc(logits, decoder_input_ids, ignore_index=tokenizer.pad_token_id)
            eval_loss += loss.item()
            eval_acc += accuracy

            progress_bar.set_postfix({f"{desc} Loss": f"{loss.item():.4f}", f"{desc} Acc": f"{accuracy:.4f}"})
    return eval_loss / len(dataloader), eval_acc / len(dataloader)


def run_model_test(model, dataloader, loss_fct, tokenizer):
    """
    Runs the model on the test set.
    """
    return evaluate(model, dataloader, loss_fct, tokenizer, desc="Testing")


def main():
    args = set_args()
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model_path)
    all_df = load_data(args.data_path)

    train_df, val_df, test_df = split_data(all_df, train_size=0.8, val_size=0.1, random_state=42)
    print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_dataset = MT5Dataset(train_df, tokenizer)
    val_dataset = MT5Dataset(val_df, tokenizer)
    test_dataset = MT5Dataset(test_df, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    model = MT5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)

    num_parameters = sum(p.numel() for p in model.parameters())
    print('Total parameters:', num_parameters)

    if torch.cuda.is_available():
        model.cuda()

    total_steps = int(len(train_dataset) * args.epochs / args.batch_size / args.gradient_accumulation)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps_rate * total_steps),
        num_training_steps=total_steps
    )

    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_val_acc = 0
    best_model_path = os.path.join(args.output_dir, "best_model_mt5_4")
    os.makedirs(best_model_path, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.epochs} ==========")
        epoch_start_time = datetime.now()

        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, scheduler, loss_fct, tokenizer, args)
        print(f"Training: Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # 验证阶段
        val_loss, val_acc = evaluate(model, val_dataloader, loss_fct, tokenizer, desc="Validation")
        print(f"Validation: Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"Best model updated at epoch {epoch + 1}")

        epoch_save_path = os.path.join(args.output_dir, f"model_epoch_{epoch + 1}")
        os.makedirs(epoch_save_path, exist_ok=True)
        model.save_pretrained(epoch_save_path)
        tokenizer.save_pretrained(epoch_save_path)

        epoch_finish_time = datetime.now()
        print(f"Epoch {epoch + 1} finished, Time cost: {epoch_finish_time - epoch_start_time}")

    print("\n========== Testing ==========")
    model = MT5ForConditionalGeneration.from_pretrained(best_model_path)
    if torch.cuda.is_available():
        model.cuda()
    test_loss, test_acc = run_model_test(model, test_dataloader, loss_fct, tokenizer)
    print(f"Test: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()
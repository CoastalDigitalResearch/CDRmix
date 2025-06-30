import os
import torch
from torch.utils.data import DataLoader
from cdrmix.cdrmix_model import RWKVMoEModel
from cdrmix.data_loader import CDRDataset
from utils.config import load_yaml_config
from utils.tokenizer_utils import load_tokenizer
from tqdm import tqdm
import argparse

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            logits, _ = model(inputs, None)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return avg_loss, ppl.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/eval-1b.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    tokenizer = load_tokenizer(config["tokenizer_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model init
    model = RWKVMoEModel(**config["model"])
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    # Dataset
    eval_dataset = CDRDataset(config["dataset"])
    dataloader = DataLoader(eval_dataset, batch_size=config.get("batch_size", 4))

    # Eval
    loss, ppl = evaluate(model, dataloader, device)
    print(f"Eval loss: {loss:.4f}, Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import get_scheduler
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import json
import argparse


class EarlyStopping:
    def __init__(self, patience=1, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early Stopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def format_data(examples):
    model_inputs = []
    model_targets = []
    for episode in range(len(examples)):
        sequence = f"<relationship> {examples[episode]['relationship']}"
        for session in range(5):
            if session > 0:
                prefix = "{} <{}> {}".format(sequence, examples[episode]['time_interval'][session],
                                             examples[episode]['summary'][session - 1])
            else:
                prefix = sequence

            input_sequence = "{} [now]".format(prefix)

            for utterance in range(len(examples[episode]['conversation'][session])):
                input_sequence = "{} <{}>".format(input_sequence, examples[episode]['speakers'][session][utterance])
                model_inputs.append(input_sequence)
                target_text = examples[episode]['conversation'][session][utterance]
                model_targets.append(target_text)
                input_sequence = "{} {}".format(input_sequence, target_text)

            input_sequence = "{} <{}>".format(input_sequence, examples[episode]['speakers'][session][utterance - 1])
            model_inputs.append(input_sequence)
            model_targets.append("[END]")

            sequence = prefix

    data = {'inputs': model_inputs, 'targets': model_targets}
    data = Dataset.from_dict(data)

    return data


class CCDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, inputs_len, target_len):
        self.tokenizer = tokenizer
        self.inputs_len = inputs_len
        self.target_len = target_len
        self.source_text = data['inputs']
        self.target_text = data['targets']

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        model_inputs = self.source_text[index]
        model_targets = self.target_text[index]

        inputs = self.tokenizer.batch_encode_plus(
            [model_inputs],
            max_length=self.inputs_len,
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer.batch_encode_plus(
            [model_targets],
            max_length=self.target_len,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = targets["input_ids"].squeeze()

        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long),
            "labels": labels.to(dtype=torch.long),
        }


def main(args):
    set_seed(777)
    accelerator = Accelerator()
    device = accelerator.device
    logger = get_logger(__name__, log_level="INFO")

    with open(args.train_path, 'r') as f:
        sample = [json.loads(line) for line in f]
    train = format_data(sample)
    del sample

    with open(args.valid_path, 'r') as f:
        sample = [json.loads(line) for line in f]
    valid = format_data(sample)
    del sample

    checkpoint = "facebook/bart-large"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = model.to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    max_input_len = args.max_input_len
    max_target_len = args.max_target_len
    batch_size = args.batch_size

    train_dataset = CCDataset(train, tokenizer, max_input_len, max_target_len)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    valid_dataset = CCDataset(valid, tokenizer, max_input_len, max_target_len)
    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    num_train_epochs = args.epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(model, optimizer,
                                                                                             train_dataloader,
                                                                                             valid_dataloader,
                                                                                             lr_scheduler)
    early_stopping = EarlyStopping(patience=1, verbose=True)

    result = {}
    for epoch in tqdm(range(num_train_epochs)):
        model.train()
        train_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                train_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        valid_loss = 0
        for step, batch in enumerate(valid_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                valid_loss += loss.detach().float()

        result["train_loss"] = train_loss.item() / len(train_dataloader)
        result["valid_loss"] = valid_loss.item() / len(valid_dataloader)
        result["epoch"] = epoch + 1
        logger.info(result)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f'{args.save_path}/epoch_{epoch + 1}/', is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(f'{args.save_path}/epoch_{epoch + 1}/')

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            logger.info("Early Stopping!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="train.jsonl")
    parser.add_argument("--valid_path", type=str, default="valid.jsonl")
    parser.add_argument("--max_input_len", type=int, default=1024)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_path", type=str, default="./generation_module")

    args = parser.parse_args()

    main(args)

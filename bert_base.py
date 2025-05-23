import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
from functools import partial
import random
import yaml

def set_seed(seed):
    """Set the random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def load_config_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def tokenize_func(examples, tokenizer, max_length):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )

def compute_metrics(pred: EvalPrediction):
    """
    Compute metrics for evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config/bert_base_sst2.yaml", help="Path to YAML config file")
    config_args, remaining_args = config_parser.parse_known_args()    
    config = {} 
    if config_args.config:
        config = load_config_yaml(config_args.config)

    parser = argparse.ArgumentParser(description="Train BERT base model for sequence classification")
    parser.add_argument("--model_name_or_path", type=str, default=config.get("model_name_or_path", "google-bert/bert-base-uncased"), help="Model checkpoint name or path")
    parser.add_argument("--task", type=str, default=config.get("task", "stanfordnlp/sst2"), help="Dataset task to train on")
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 16), help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=config.get("max_length", 512), help="Maximum sequence length")
    parser.add_argument('--seed', type=int, default=config.get("seed", 42), help='Specify random seed, default -1')
    parser.add_argument("--output_dir", type=str, default=config.get("output_dir", "./results/bert_base_sst2"))
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 10))
    parser.add_argument("--learning_rate", type=float, default=config.get("learning_rate", 5e-5))
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 0.01))
    parser.add_argument("--warmup_steps", type=int, default=config.get("warmup_steps", 500))
    parser.add_argument("--logging_dir", type=str, default=config.get("logging_dir", "./logs"))
    parser.add_argument("--logging_steps", type=int, default=config.get("logging_steps", 100))
    parser.add_argument("--eval_steps", type=int, default=config.get("eval_steps", 500))
    parser.add_argument("--save_steps", type=int, default=config.get("save_steps", 500))
    parser.add_argument("--fp16", type=bool, default=config.get("fp16", False))
    
    args = parser.parse_args(remaining_args)

    for arg in vars(args):
        val = getattr(args, arg)
        if isinstance(val, str) and val.lower() in {"none", "true", "false"}:
            setattr(args, arg, {"none": None, "true": True, "false": False}[val.lower()])
    return args


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "parser_para.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    dataset = load_dataset(args.task)
    num_labels = len(dataset["train"].features["label"].names)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenize_with_args = partial(tokenize_func, tokenizer=tokenizer, max_length=args.max_length)
    tokenized_datasets = dataset.map(tokenize_with_args, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset, val_dataset = tokenized_datasets["train"], tokenized_datasets["validation"]

    # Load standard BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*2,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=args.fp16,  # Mixed precision training
        save_total_limit=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Training the model...")
    trainer.train()
    
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save evaluation results to txt file
    eval_results_path = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(eval_results_path, "w") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
    print(f"Evaluation results saved to: {eval_results_path}")
    
    print("Saving the model...")
    # Save final model to output_dir
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
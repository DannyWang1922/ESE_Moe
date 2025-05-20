import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    EvalPrediction,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse

# Import the custom BertMoE model
from models.modeling_bert_moe import BertMoEForSequenceClassification, BertMoEConfig
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
    config_parser.add_argument("--config", type=str, default="config/bert_moe.yaml", help="Path to YAML config file")
    config_args, remaining_args = config_parser.parse_known_args()    
    config = {} 
    if config_args.config:
        config = load_config_yaml(config_args.config)

    parser = argparse.ArgumentParser(description="Evaluate a causal LM on MTEB")
    parser.add_argument("--model_name", type=str, default=config.get("model_name_or_path", "google-bert/bert-base-uncased"), help="Model checkpoint name or path")
    parser.add_argument("--task", type=str, default=config.get("task", "STS"), help="MTEB task to evaluate on")
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 32), help="Batch size for encoding")
    parser.add_argument("--max_length", type=int, default=config.get("max_length", 512), help="Batch size for encoding")
    parser.add_argument("--save_dir", type=str, default=config.get("output_folder", None), help="Folder to save results")
    parser.add_argument('--seed', type=int, default=config.get("seed", 42), help='Specify random seed, default -1')

    parser.add_argument("--num_experts", type=int, default=config.get("num_experts", 8))
    parser.add_argument("--top_k", type=int, default=config.get("top_k", 2))
    parser.add_argument("--expert_dropout", type=float, default=config.get("expert_dropout", 0.1))
    parser.add_argument("--router_temperature", type=float, default=config.get("router_temperature", 0.1))
    parser.add_argument("--router_noise_epsilon", type=str, default=config.get("router_noise_epsilon", "1e-2"))
    parser.add_argument("--router_training_noise", type=bool, default=config.get("router_training_noise", True))
    parser.add_argument("--use_load_balancing", type=bool, default=config.get("use_load_balancing", True))
    parser.add_argument("--router_z_loss_coef", type=str, default=config.get("router_z_loss_coef", "1e-3"))
    parser.add_argument("--router_aux_loss_coef", type=float, default=config.get("router_aux_loss_coef", 0.01))
    parser.add_argument("--track_expert_metrics", type=bool, default=config.get("track_expert_metrics", True))

    parser.add_argument("--output_dir", type=str, default=config.get("output_dir", "./results/bert_moe_sst2"))
    parser.add_argument("--num_train_epochs", type=int, default=config.get("num_train_epochs", 3))
    parser.add_argument("--per_device_train_batch_size", type=int, default=config.get("per_device_train_batch_size", 16))
    parser.add_argument("--per_device_eval_batch_size", type=int, default=config.get("per_device_eval_batch_size", 64))
    parser.add_argument("--warmup_steps", type=int, default=config.get("warmup_steps", 500))
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 0.01))
    parser.add_argument("--logging_dir", type=str, default=config.get("logging_dir", "./logs"))
    parser.add_argument("--logging_steps", type=int, default=config.get("logging_steps", 100))
    parser.add_argument("--eval_steps", type=int, default=config.get("eval_steps", 500))
    parser.add_argument("--save_steps", type=int, default=config.get("save_steps", 500))
    parser.add_argument("--fp16", type=bool, default=config.get("fp16", True))
    
    args = parser.parse_args(remaining_args)

    args.save_dir =  f"mteb_results/{args.model_name.split('/')[-1]}"
    for arg in vars(args):
        val = getattr(args, arg)
        if isinstance(val, str) and val.lower() in {"none", "true", "false"}:
            setattr(args, arg, {"none": None, "true": True, "false": False}[val.lower()])
    return args


def main(args):
    dataset = load_dataset("stanfordnlp/sst2")
    num_labels = len(dataset["train"].features["label"].names)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenize_with_args = partial(tokenize_func, tokenizer=tokenizer, max_length=args.max_length)
    tokenized_datasets = dataset.map(tokenize_with_args, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset, val_dataset = tokenized_datasets["train"], tokenized_datasets["validation"]

    moe_config = BertMoEConfig(
        num_labels=num_labels,  # Binary classification for SST-2
        # MoE-specific parameters
        num_experts=args.num_experts,
        top_k=args.top_k,
        expert_dropout=args.expert_dropout,
        router_temperature=args.router_temperature,
        router_noise_epsilon=args.router_noise_epsilon,
        router_training_noise=args.router_training_noise,
        use_load_balancing=args.use_load_balancing,
        router_z_loss_coef=args.router_z_loss_coef,
        router_aux_loss_coef= args.router_aux_loss_coef,
        track_expert_metrics=args.track_expert_metrics,
    )
    
    # Initialize BertMoE model with configuration
    model = BertMoEForSequenceClassification(moe_config)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,  # Mixed precision training
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    print("Training the model...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the model
    print("Saving the model...")
    trainer.save_model("./bert_moe_sst2_final")
    tokenizer.save_pretrained("./bert_moe_sst2_final")
    
    # Inference example
    test_sentence = "This movie was really enjoyable and well-acted."
    inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    print(f"Test sentence: '{test_sentence}'")
    print(f"Predicted class: {predicted_class} ({'positive' if predicted_class == 1 else 'negative'})")
    
    # Analyze expert utilization (if metrics are tracked)
    if hasattr(model.bert.encoder, "expert_metrics"):
        print("\nExpert Utilization Metrics:")
        expert_metrics = model.bert.encoder.expert_metrics
        for key, value in expert_metrics.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
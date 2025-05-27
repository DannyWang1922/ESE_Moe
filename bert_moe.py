import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse

# Import the custom BertMoE model
from models.modeling_bert_moe import BertMoEForSequenceClassification
from models.configuration_bert_moe import BertMoEConfig
from functools import partial
import random
import yaml
from bertMoE import MoETrainer

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
    config_parser.add_argument("--config", type=str, default="config/bert_moe_sst2.yaml", help="Path to YAML config file")
    config_args, remaining_args = config_parser.parse_known_args()    
    config = {} 
    if config_args.config:
        config = load_config_yaml(config_args.config)

    parser = argparse.ArgumentParser(description="Evaluate a causal LM on MTEB")
    parser.add_argument("--model_name_or_path", type=str, default=config.get("model_name_or_path", "google-bert/bert-base-uncased"), help="Model checkpoint name or path")
    parser.add_argument("--task", type=str, default=config.get("task", "stanfordnlp/sst2"), help="Dataset task to train on")
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 16), help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=config.get("max_length", 512), help="Maximum sequence length")
    parser.add_argument('--seed', type=int, default=config.get("seed", 42), help='Specify random seed, default -1')
    parser.add_argument("--output_dir", type=str, default=config.get("output_dir", "./results/bert_moe_sst2"))
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 10))
    parser.add_argument("--learning_rate", type=float, default=config.get("learning_rate", 5e-5))
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 0.01))
    parser.add_argument("--warmup_steps", type=int, default=config.get("warmup_steps", 500))
    parser.add_argument("--logging_dir", type=str, default=config.get("logging_dir", "./logs"))
    parser.add_argument("--logging_steps", type=int, default=config.get("logging_steps", 100))
    parser.add_argument("--eval_steps", type=int, default=config.get("eval_steps", 500))
    parser.add_argument("--save_steps", type=int, default=config.get("save_steps", 500))
    parser.add_argument("--fp16", type=bool, default=config.get("fp16", False))

    # MoE specific arguments
    parser.add_argument("--num_experts", type=int, default=config.get("num_experts", 8))
    parser.add_argument("--top_k", type=int, default=config.get("top_k", 2))
    parser.add_argument("--expert_dropout", type=float, default=config.get("expert_dropout", 0.1))
    parser.add_argument("--router_temperature", type=float, default=config.get("router_temperature", 0.1))
    parser.add_argument("--router_noise_epsilon", type=float, default=config.get("router_noise_epsilon", 1e-2))
    parser.add_argument("--router_training_noise", type=bool, default=config.get("router_training_noise", True))
    parser.add_argument("--use_load_balancing", type=bool, default=config.get("use_load_balancing", True))
    parser.add_argument("--router_z_loss_coef", type=float, default=config.get("router_z_loss_coef", 1e-3))
    parser.add_argument("--router_aux_loss_coef", type=float, default=config.get("router_aux_loss_coef", 0.01))
    parser.add_argument("--track_expert_metrics", type=bool, default=config.get("track_expert_metrics", True))
    parser.add_argument("--load_balance_loss_weight", type=float, default=config.get("load_balance_loss_weight", 0.01))
    
    # Add MoE layer selection argument
    parser.add_argument("--moe_layers", type=str, default=config.get("moe_layers", "[11]"),
                        help="""Specify which layers to use MoE. Options: - 'all' (default) - '[0,2,4,6]': Use MoE for specific layers"""
    )
    
    args = parser.parse_args(remaining_args)

    for arg in vars(args):
        val = getattr(args, arg)
        if isinstance(val, str) and val.lower() in {"none", "true", "false"}:
            setattr(args, arg, {"none": None, "true": True, "false": False}[val.lower()])
    return args


def process_moe_layers_arg(moe_layers_str):
    """
    Process the moe_layers argument into a format the model can use.
    
    Args:
        moe_layers_str: String specifying which layers to use MoE
        
    Returns:
        Processed moe_layers value ('all' or list of integers)
    """
    if moe_layers_str == 'all':
        return 'all'
    else:
        try:
            import ast
            moe_layers_list = ast.literal_eval(moe_layers_str)
            if isinstance(moe_layers_list, list):
                # Validate all elements are integers
                for idx in moe_layers_list:
                    if not isinstance(idx, int):
                        raise ValueError(f"All layer indices must be integers, got {type(idx)}")
                return moe_layers_list
            else:
                raise ValueError(f"moe_layers must be 'all' or a list of integers")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid moe_layers format: {moe_layers_str}. Must be 'all' or a list like '[0,2,4,6]'") from e


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

    # Process moe_layers argument
    processed_moe_layers = process_moe_layers_arg(args.moe_layers)
    
    moe_config = BertMoEConfig(
        num_labels=num_labels, 
        num_experts=args.num_experts,
        top_k=args.top_k,
        expert_dropout=args.expert_dropout,
        router_temperature=args.router_temperature,
        router_noise_epsilon=args.router_noise_epsilon,
        router_training_noise=args.router_training_noise,
        use_load_balancing=args.use_load_balancing,
        router_z_loss_coef=args.router_z_loss_coef,
        router_aux_loss_coef=args.router_aux_loss_coef,
        track_expert_metrics=args.track_expert_metrics,
        moe_layers=processed_moe_layers,  # Add the MoE layers configuration
    )
    model = BertMoEForSequenceClassification(moe_config)
    
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
    
    # trainer = MoETrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     compute_metrics=compute_metrics,
    #     tokenizer=tokenizer,
    #     load_balance_loss_weight=args.load_balance_loss_weight,  # Pass the loss weight
    # )
    
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
    
    # Analyze expert utilization (if metrics are tracked)
    if hasattr(model.bert_moe.encoder, "expert_metrics"):
        print("\nExpert Utilization Metrics:")
        expert_metrics = model.bert_moe.encoder.expert_metrics
        
        # Save expert metrics to txt file
        expert_metrics_path = os.path.join(args.output_dir, "expert_utilization_metrics.txt")
        with open(expert_metrics_path, "w") as f:
            f.write("Expert Utilization Metrics\n")
            f.write("=========================\n\n")
            
            # Include which layers used MoE
            if "moe_layers_used" in expert_metrics:
                f.write(f"MoE layers used: {expert_metrics['moe_layers_used']}\n")
                print(f"MoE layers used: {expert_metrics['moe_layers_used']}")
            
            for key, value in expert_metrics.items():
                if key != "moe_layers_used":  # Avoid duplicate printing
                    print(f"{key}: {value}")
                    f.write(f"{key}: {value}\n")
        print(f"Expert metrics saved to: {expert_metrics_path}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
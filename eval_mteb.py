import argparse
import json
import os
import mteb
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from ese_moe import ESE_Moe
import random
import numpy as np

def set_seed(seed):
    """Set the random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def load_config_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config/olmoe.yaml", help="Path to YAML config file")
    config_args, remaining_args = config_parser.parse_known_args()    
    config = {} 
    if config_args.config:
        config = load_config_yaml(config_args.config)

    parser = argparse.ArgumentParser(description="Evaluate a causal LM on MTEB")
    parser.add_argument("--model_name", type=str, default=config.get("model_name_or_path", "allenai/OLMoE-1B-7B-0924"), help="Model checkpoint name or path")
    parser.add_argument("--tasks", type=str, default=config.get("tasks", ["STS"]), help="MTEB tasks to evaluate on")
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 32), help="Batch size for encoding")
    parser.add_argument("--max_length", type=int, default=config.get("max_length", 512), help="Batch size for encoding")
    parser.add_argument("--save_dir", type=str, default=config.get("output_folder", None), help="Folder to save results")
    parser.add_argument('--seed', type=int, default=42, help='Specify random seed, default -1')
    parser.add_argument('--use_8_bit', type=str, default="true", help='Use 8-bit quantization, default false')
    
    args = parser.parse_args(remaining_args)

    args.save_dir =  f"mteb_results/{args.model_name.split('/')[-1]}"
    
    for arg in vars(args):
        val = getattr(args, arg)
        if isinstance(val, str) and val.lower() in {"none", "true", "false"}:
            setattr(args, arg, {"none": None, "true": True, "false": False}[val.lower()])
    return args


def main(args):
    # save the running configuration
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "parser_para.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ESE_Moe(model_name_or_path=args.model_name, 
                    device= device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    use_8_bit=args.use_8_bit,)

    if args.tasks == 'all':
        benchmark = mteb.get_benchmark("MTEB(eng, v1)")
        evaluation = MTEB(tasks=benchmark )
    elif 'STS' in args.tasks:
        STS_TASKS = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICK-R']
        tasks= mteb.get_tasks(tasks=STS_TASKS)
        evaluation = MTEB(tasks=tasks)

    evaluation.run(model, output_folder=args.save_dir, encode_kwargs={"batch_size": args.batch_size})
    print(f"Evaluation complete. Results saved to: {args.save_dir}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)

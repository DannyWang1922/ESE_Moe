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
from prettytable import PrettyTable

def set_seed(seed):
    """Set the random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def load_config_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def extract_task_scores(eval_res):
    task_names = []
    main_scores = []
    
    for result in eval_res:
        task_name = result.task_name
        main_score = result.scores["test"][0]["main_score"]
        
        task_names.append(task_name)
        main_scores.append("%.2f" %main_score)
    
    return task_names, main_scores

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config/eval_olmoe.yaml", help="Path to YAML config file")
    config_args, remaining_args = config_parser.parse_known_args()    
    config = {} 
    if config_args.config:
        config = load_config_yaml(config_args.config)

    parser = argparse.ArgumentParser(description="Evaluate a causal LM on MTEB")
    parser.add_argument("--model_name", type=str, default=config.get("model_name_or_path", "allenai/OLMoE-1B-7B-0924"), help="Model checkpoint name or path")
    parser.add_argument("--tasks", type=str, default=config.get("tasks", "STS"), help="MTEB tasks to evaluate on")
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 32), help="Batch size for encoding")
    parser.add_argument("--max_length", type=int, default=config.get("max_length", 512), help="Batch size for encoding")
    parser.add_argument("--save_dir", type=str, default=config.get("output_folder", None), help="Folder to save results")
    parser.add_argument('--seed', type=int, default=config.get("seed", 42), help='Specify random seed, default -1')
    parser.add_argument('--use_8_bit', type=str, default=config.get("use_8_bit", "true"), help='Use 8-bit quantization, default false')
    
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

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    model = ESE_Moe(model_name_or_path=args.model_name, 
                    device= device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    use_8_bit=args.use_8_bit,)

    if args.tasks == 'all':
        benchmark = mteb.get_benchmark("MTEB(eng, v1)")
        evaluation = MTEB(tasks=benchmark )
    elif args.tasks == 'STS':
        STS_TASKS = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICK-R']
        tasks= mteb.get_tasks(tasks=STS_TASKS)
        evaluation = MTEB(tasks=tasks)

    eval_res = evaluation.run(model, output_folder=args.save_dir, encode_kwargs={"batch_size": args.batch_size})
    task_list, score_list = extract_task_scores(eval_res)
    
    pavg = sum([float(s) for s in score_list]) / len(score_list)
    score_list.append("%.2f" % pavg)
    task_list.append("Avg.")
    task_list.insert(0, "Model_Name")
    score_list.insert(0, args.model_name.split('/')[-1])

    print_table(task_list, score_list)
    print(f"Evaluation complete. Results saved to: {args.save_dir}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)

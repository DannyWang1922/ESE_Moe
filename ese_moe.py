from typing import Dict, List, Union
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from models.modeling_olmoe import OlmoeForCausalLM
import os
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ESE_Moe')

def load_pretrained_model(model_name, model_type, device, use_8_bit) -> tuple:
    """Loads a pretrained model from HuggingFace.

    Args:
        model_name (str): Name of model (e.g. "mistralai/Mistral-7B-v0.1")
        model_type (str): Type of model to load ("deepseek-moe", "Qwen", "OLMoE")
        device (str): Device to load the model on (e.g. "cuda", "cpu")

    Returns:
        tuple(model, tokenizer): Loaded model and tokenizer
    """
    use_quantization = (device.type == "cuda" and use_8_bit)
    logger.info("use_quantization: %s", use_quantization)
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_type == 'OLMoE':
        if use_quantization:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = OlmoeForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            model = OlmoeForCausalLM.from_pretrained(model_name).bfloat16().to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, tokenizer


class ESE_Moe(torch.nn.Module):
    def __init__(self, 
                 model_name_or_path: str, 
                 device: str,
                 max_length: int,
                 batch_size: int,
                 use_8_bit: bool,
                 **kwargs: Any):
        super().__init__()
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_8_bit = use_8_bit

        if 'OLMoE' in model_name_or_path:
            self.model, self.tokenizer = load_pretrained_model(model_name_or_path, "OLMoE",  device=self.device, use_8_bit=self.use_8_bit)
        else: 
            raise ValueError(f"Unknown model source in path: {model_name_or_path}")

    @torch.no_grad()
    def encode(self, 
               sentences: Union[str, List[str]],
               **kwargs 
    ) -> np.ndarray:
        batch_size = self.batch_size
        max_length = self.max_length
        
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for start_idx in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
            batch = sentences[start_idx:start_idx + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length
            ).to(self.device)

            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
            last_token_embedding = last_hidden_state[:, -1, :]  # (batch_size, hidden_dim)

            embeddings.append(last_token_embedding.cpu())

        return torch.cat(embeddings, dim=0).numpy()
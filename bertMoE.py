"""
Fixed Custom Trainer for BertMoE with load balancing loss support.
Properly handles the num_items_in_batch parameter and learning_rate parameter.
"""

import torch
from torch import nn
from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Dict, Union, Any, Optional, Tuple


class MoETrainer(Trainer):
    """
    Custom Trainer that handles auxiliary losses from MoE models.
    
    This trainer extends the base Trainer to:
    1. Collect load balancing losses from all MoE layers
    2. Aggregate them with the main task loss
    3. Track and log expert utilization metrics
    """
    
    def __init__(self, *args, load_balance_loss_weight: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_balance_loss_weight = load_balance_loss_weight
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the total loss including auxiliary MoE losses.
        
        Args:
            model: The model being trained
            inputs: Input batch
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in the batch (required by transformers>=4.26.0)
            
        Returns:
            loss: Combined loss (task loss + auxiliary losses)
            outputs: Model outputs (if return_outputs=True)
        """
        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("Labels must be provided for training")
        
        # Forward pass
        outputs = model(**inputs)
        
        # Get the main task loss
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            loss = outputs[0]
        else:
            raise ValueError("Model must return a loss")
        
        # Collect load balancing losses from all MoE layers
        load_balance_loss = self._collect_load_balance_losses(model)
        
        # Combine losses
        if load_balance_loss is not None and load_balance_loss > 0:
            total_loss = loss + self.load_balance_loss_weight * load_balance_loss
            
            # Log the individual loss components
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "loss/task": loss.detach().item() if hasattr(loss, 'detach') else float(loss),
                    "loss/load_balance": load_balance_loss.detach().item() if hasattr(load_balance_loss, 'detach') else float(load_balance_loss),
                    "loss/total": total_loss.detach().item() if hasattr(total_loss, 'detach') else float(total_loss),
                })
        else:
            total_loss = loss
            
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _collect_load_balance_losses(self, model) -> Optional[torch.Tensor]:
        """
        Collect load balancing losses from all MoE layers in the model.
        
        Args:
            model: The BertMoE model
            
        Returns:
            aggregated_loss: Sum of all load balancing losses
        """
        total_loss = None
        num_layers = 0
        
        # Access the encoder from the model
        if hasattr(model, 'bert_moe'):
            encoder = model.bert_moe.encoder
        elif hasattr(model, 'encoder'):
            encoder = model.encoder
        else:
            return None
            
        # Iterate through all layers
        for layer_idx, layer in enumerate(encoder.layer):
            if hasattr(layer, 'router') and hasattr(layer, '_last_router_probs') and hasattr(layer, '_last_router_indices'):
                # Get the stored router logits and indices from the last forward pass
                router_logits = layer._last_router_probs  # This should be the full logits
                router_indices = layer._last_router_indices
                
                if router_logits is not None and router_indices is not None:
                    # Compute load balancing loss for this layer
                    # Note: We pass logits (not probabilities) to compute_load_balancing_loss
                    layer_loss = layer.router.compute_load_balancing_loss(router_logits, router_indices)
                    
                    if total_loss is None:
                        total_loss = layer_loss
                    else:
                        total_loss = total_loss + layer_loss
                    num_layers += 1
                    
        # Average across layers
        if num_layers > 0 and total_loss is not None:
            return total_loss / num_layers
        else:
            return None
    
    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        """
        Override to add expert utilization metrics logging.
        Using *args and **kwargs for compatibility with different transformers versions.
        """
        # Call parent method first with all arguments
        super()._maybe_log_save_evaluate(tr_loss, *args, **kwargs)
        
        # Extract model from args for expert metrics logging
        # Position of model in args depends on transformers version
        # Try to find model in args
        model = None
        if len(args) >= 2:
            # In most versions, model is the third argument (index 1 in args after tr_loss)
            potential_model = args[1]
            # Check if it's actually a model by checking for expected attributes
            if hasattr(potential_model, 'bert_moe') or hasattr(potential_model, 'encoder'):
                model = potential_model
        
        # If not found in expected position, search through args
        if model is None:
            for arg in args:
                if hasattr(arg, 'bert_moe') or hasattr(arg, 'encoder'):
                    model = arg
                    break
        
        # Log expert utilization metrics
        if model and self.state.global_step % self.args.logging_steps == 0:
            self._log_expert_metrics(model)
    
    def _log_expert_metrics(self, model):
        """
        Log expert utilization metrics from the model.
        """
        # Access the encoder
        if hasattr(model, 'bert_moe'):
            encoder = model.bert_moe.encoder
        elif hasattr(model, 'encoder'):
            encoder = model.encoder
        else:
            return
            
        # Log overall expert utilization if available
        if hasattr(encoder, 'expert_metrics'):
            metrics = encoder.expert_metrics
            
            # # Log expert utilization
            # if 'expert_utilization' in metrics:
            #     utilization = metrics['expert_utilization']
            #     if isinstance(utilization, torch.Tensor):
            #         # Log per-expert utilization
            #         for expert_idx in range(len(utilization)):
            #             self.log({f"experts/utilization_expert_{expert_idx}": utilization[expert_idx].item()})
                    
            #         # Log overall statistics
            #         self.log({
            #             "experts/utilization_mean": utilization.mean().item(),
            #             "experts/utilization_std": utilization.std().item(),
            #             "experts/utilization_min": utilization.min().item(),
            #             "experts/utilization_max": utilization.max().item(),
            #         })
            
            # Log load balance metric
            if 'expert_load_balance' in metrics:
                load_balance_value = metrics['expert_load_balance'].item() if hasattr(metrics['expert_load_balance'], 'item') else float(metrics['expert_load_balance'])
                self.log({"experts/load_balance_cv": load_balance_value})
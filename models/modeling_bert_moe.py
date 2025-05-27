"""PyTorch BERT model."""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertEmbeddings,
    BertPooler    
)
from transformers.models.bert.modeling_bert import *

logger = logging.get_logger(__name__)


class BertMoEExpert(nn.Module):
    """
    Expert module for Mixture of Experts BERT model.
    
    Each expert is a feed-forward network similar to the original BERT feed-forward,
    consisting of an intermediate layer followed by an output layer.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create expert's intermediate layer (equivalent to BertIntermediate)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # Activation function
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        
        # Create expert's output layer (equivalent to BertOutput but without LayerNorm/residual)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize expert weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize the expert weights.
        Using specific initialization can help with training stability.
        """
        # Initialize intermediate layer
        nn.init.normal_(self.intermediate.weight, mean=0.0, std=self.config.initializer_range)
        if self.intermediate.bias is not None:
            nn.init.zeros_(self.intermediate.bias)
        
        # Initialize output layer
        nn.init.normal_(self.output.weight, mean=0.0, std=self.config.initializer_range)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)
    
    def forward(self, hidden_states):
        """
        Forward pass through the expert.
        
        Args:
            hidden_states: Tensor of shape [batch_size, hidden_size]
                Token representations to process.
                
        Returns:
            output: Tensor of shape [batch_size, hidden_size]
                Processed token representations.
        """
        # Forward through intermediate layer
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        
        # Forward through output layer
        output = self.output(intermediate_output)
        output = self.dropout(output)
        
        return output


class BertMoEExpertPool(nn.Module):
    """
    Pool of experts for Mixture of Experts BERT model.
    
    This module manages a collection of expert networks and provides methods
    to access them and compute outputs in parallel when possible.
    """
    
    def __init__(self, config, num_experts=8):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        
        # Create a ModuleList of experts
        self.experts = nn.ModuleList([BertMoEExpert(config) for _ in range(num_experts)])
        
        # Expert initialization strategy
        self.expert_init_strategy = config.expert_init_strategy
        
        # Apply initialization strategy
        if self.expert_init_strategy == "diverse":
            # Initialize experts with different parameters to encourage specialization
            self._initialize_diverse_experts()
        
        # Optional expert dropout for regularization
        self.expert_dropout = config.expert_dropout
        
        # Flag for parallelizing expert computation when possible
        self.parallel_computation = config.parallel_expert_computation
    
    def _initialize_diverse_experts(self):
        """
        Initialize experts with diverse parameters to encourage specialization.
        
        This can help experts develop different capabilities by giving them
        different starting points.
        """
        for i, expert in enumerate(self.experts):
            # Adjust weights slightly for each expert to break symmetry
            with torch.no_grad():
                # Scale intermediate weights by a small factor based on expert index
                scale_factor = 1.0 + (i - self.num_experts // 2) * 0.01
                expert.intermediate.weight.data *= scale_factor
                
                # Optionally, slightly adjust bias terms too
                if expert.intermediate.bias is not None:
                    expert.intermediate.bias.data += (i - self.num_experts // 2) * 0.001
    
    def forward_expert(self, expert_idx, hidden_states):
        """
        Forward pass through a specific expert.
        
        Args:
            expert_idx: Integer index of the expert to use.
            hidden_states: Tensor of shape [batch_size, hidden_size]
                Token representations to process.
                
        Returns:
            output: Tensor of shape [batch_size, hidden_size]
                Processed token representations.
        """
        # Get the expert
        expert = self.experts[expert_idx]
        
        # Apply expert dropout during training (randomly disable experts)
        if self.training and self.expert_dropout > 0 and torch.rand(1).item() < self.expert_dropout:
            # If expert is dropped, return zeros or the input (here we choose zeros)
            return torch.zeros_like(hidden_states)
        
        # Forward through the expert
        return expert(hidden_states)
    
    def forward_all_experts(self, hidden_states):
        """
        Forward pass through all experts in parallel.
        
        This method is useful when we want to compute outputs for all experts
        at once, which can be more efficient on certain hardware.
        
        Args:
            hidden_states: Tensor of shape [batch_size, hidden_size]
                Token representations to process.
                
        Returns:
            expert_outputs: Tensor of shape [num_experts, batch_size, hidden_size]
                Outputs from all experts.
        """
        if not self.parallel_computation:
            # Sequential computation
            expert_outputs = []
            for expert_idx in range(self.num_experts):
                expert_output = self.forward_expert(expert_idx, hidden_states)
                expert_outputs.append(expert_output)
            return torch.stack(expert_outputs)
        
        # Parallel computation
        # This can be more efficient on GPUs but may use more memory
        batch_size = hidden_states.shape[0]
        
        # Repeat inputs for each expert
        # Shape: [num_experts * batch_size, hidden_size]
        expanded_inputs = hidden_states.repeat(self.num_experts, 1)
        
        # Create an expert assignment tensor to track which expert processes which example
        # Shape: [num_experts * batch_size]
        expert_indices = torch.arange(self.num_experts, device=hidden_states.device).repeat_interleave(batch_size)
        
        # Process all inputs through all experts in a single pass
        all_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            # Select inputs for this expert
            mask = (expert_indices == expert_idx)
            if mask.any():
                expert_inputs = expanded_inputs[mask]
                expert_output = expert(expert_inputs)
                all_outputs.append(expert_output)
            else:
                # No inputs for this expert
                all_outputs.append(torch.zeros(0, hidden_states.shape[-1], device=hidden_states.device))
        
        # Reshape to [num_experts, batch_size, hidden_size]
        expert_outputs = torch.stack([output for output in all_outputs])
        
        return expert_outputs
    
    
class BertMoERouter(nn.Module):
    """
    Router module for Mixture of Experts BERT model.
    
    This module determines which experts should process each token by computing
    routing probabilities. It takes token representations as input and outputs
    logits for each expert, which are then used to select the top-k experts.
    """
    
    def __init__(self, config, num_experts=8, top_k=2):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router projection to calculate expert routing logits
        # Input: hidden_size, Output: num_experts
        self.router_weights = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # Initialize router weights - using specific initialization can help with training stability
        # Here we use Kaiming initialization (aka He initialization)
        nn.init.kaiming_uniform_(self.router_weights.weight, a=math.sqrt(5))
        
        # Optional: add noise to encourage exploration of different experts
        self.noise_epsilon = config.router_noise_epsilon
        self.training_noise = config.router_training_noise
        
        # Optional: Temperature for softmax to control sharpness of expert selection
        self.temperature = config.router_temperature
        
        # Optional: Load balancing parameters
        self.use_load_balancing = config.use_load_balancing
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef

    def forward(self, hidden_states):
        """
        Calculate routing probabilities for each token to each expert.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
                Token representations from which to compute routing probabilities.
                
        Returns:
            router_logits: Tensor of shape [batch_size, seq_len, num_experts]
                Logits for routing each token to each expert.
        """
        # Calculate router logits [batch_size, seq_len, num_experts]
        router_logits = self.router_weights(hidden_states)
        
        # Optionally add noise during training to encourage exploration
        if self.training and self.training_noise and self.noise_epsilon > 0:
            router_noise = torch.randn_like(router_logits) * self.noise_epsilon
            router_logits = router_logits + router_noise
        
        # Apply temperature scaling if not 1.0
        if self.temperature != 1.0:
            router_logits = router_logits / self.temperature
            
        return router_logits


    def compute_load_balancing_loss(self, router_logits, router_indices):
        """
        Compute auxiliary load balancing loss to encourage equal expert utilization.
        
        This loss encourages a uniform distribution of tokens across experts,
        which helps prevent the "rich get richer" phenomenon where certain experts
        receive most of the tokens.
        
        Args:
            router_logits: Tensor of shape [batch_size, seq_len, num_experts]
                Full logits for routing each token to each expert.
            router_indices: Tensor of shape [batch_size, seq_len, top_k]
                Indices of selected top-k experts for each token.
                
        Returns:
            load_balancing_loss: Scalar tensor with the load balancing loss.
        """
        if not self.use_load_balancing:
            return torch.tensor(0.0, device=router_logits.device)
        
        batch_size, seq_len, _ = router_logits.shape
        
        # 1. Calculate the fraction of tokens assigned to each expert
        # Create one-hot encoding for expert assignments
        # Shape: [batch_size * seq_len * top_k, num_experts]
        expert_mask = torch.nn.functional.one_hot(
            router_indices.reshape(-1), 
            num_classes=self.num_experts
        ).float()
        
        # Sum across all token-expert assignments
        # Shape: [num_experts]
        tokens_per_expert = expert_mask.sum(dim=0) / (batch_size * seq_len * self.top_k)
        
        # 2. Calculate the average probability assigned to each expert
        # First, get the probabilities from logits
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # Flatten for easier manipulation
        # Shape: [batch_size * seq_len, num_experts]
        router_probs_flat = router_probs.reshape(-1, self.num_experts)
        
        # For load balancing, we care about the total probability mass assigned to each expert
        # Sum probabilities across all tokens for each expert
        # Shape: [num_experts]
        mean_expert_probs = router_probs_flat.mean(dim=0)
        
        # 3. Calculate load balancing loss using coefficient of variation
        # This encourages both uniform token distribution and uniform probability distribution
        
        # Token distribution CV (coefficient of variation)
        tokens_cv = tokens_per_expert.std() / (tokens_per_expert.mean() + 1e-10)
        
        # Probability distribution CV
        probs_cv = mean_expert_probs.std() / (mean_expert_probs.mean() + 1e-10)
        
        # Combine both aspects of load balancing
        load_balancing_loss = tokens_cv + probs_cv
        
        # Optional: Add auxiliary z-loss for numerical stability
        if hasattr(self, 'router_z_loss_coef') and self.router_z_loss_coef > 0:
            # Z-loss encourages logits to be small to improve stability
            # z_loss = log(sum(exp(logits)))^2
            log_z = torch.logsumexp(router_logits, dim=-1)
            z_loss = torch.mean(log_z ** 2)
            load_balancing_loss = load_balancing_loss + self.router_z_loss_coef * z_loss
        
        return load_balancing_loss
        

        
    def get_routing_weights(self, router_logits):
        """
        Calculate routing weights and indices for the top-k experts.
        
        Args:
            router_logits: Tensor of shape [batch_size, seq_len, num_experts]
                Logits for routing each token to each expert.
                
        Returns:
            router_probs: Tensor of shape [batch_size, seq_len, top_k]
                Probabilities for routing each token to its top-k experts.
            router_indices: Tensor of shape [batch_size, seq_len, top_k]
                Indices of the top-k experts for each token.
        """
        # Get the top-k experts for each token
        router_probs, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Normalize the routing probabilities for the selected experts
        router_probs = torch.softmax(router_probs, dim=-1)
        
        return router_probs, router_indices
    

class BertMoELayer(nn.Module):
    def __init__(self, config, num_experts=8, top_k=2):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        
        # Instead of a single intermediate and output layer, we use a mixture of experts
        # Create the router for selecting experts
        self.router = BertMoERouter(config, self.num_experts, self.top_k)
        
        # Create pool of experts (each expert is a feed-forward network)
        self.expert_pool = BertMoEExpertPool(config, self.num_experts)
        
        # Add a layer norm after MoE, similar to original BERT output layer
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # For tracking expert usage statistics
        self.expert_metrics = {
            "expert_utilization": torch.zeros(self.num_experts),
        }

        # Add these attributes to store router outputs
        self._last_router_probs = None
        self._last_router_indices = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        
        # MoE feed-forward part (replacing the original feed_forward_chunk)
        layer_output = self.moe_feed_forward(attention_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def moe_feed_forward(self, attention_output):
        """
        Mixture of Experts feed-forward network that replaces the original feed_forward_chunk.
        
        This method:
        1. Gets routing probabilities from the router
        2. Selects top-k experts for each token
        3. Dispatches tokens to their assigned experts
        4. Combines outputs according to routing weights
        5. Applies dropout and residual connection
        """
        batch_size, seq_len, hidden_size = attention_output.shape
        
        # Get routing probabilities and indices from router
        # Shape: router_probs [batch_size, seq_len, top_k], router_indices [batch_size, seq_len, top_k]
        router_logits = self.router(attention_output)
        
        # Select top-k experts and their probabilities
        router_probs, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_probs_normalized = torch.softmax(router_probs, dim=-1)

        # Store for load balancing loss computation
        self._last_router_probs = router_logits  # Store full logits, not just top-k
        self._last_router_indices = router_indices
        
        # Prepare inputs for experts
        # Reshape inputs to [batch_size * seq_len, hidden_size]
        flat_input = attention_output.reshape(-1, hidden_size)
        
        # Create tensor to collect outputs from all experts
        expert_outputs = torch.zeros(
            batch_size * seq_len, 
            hidden_size, 
            device=attention_output.device,
            dtype=attention_output.dtype
        )
        
        # Track which experts are used
        expert_counts = torch.zeros(self.num_experts, device=attention_output.device)
        
        # For each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens are routed to this expert
            # Create a mask [batch_size, seq_len, top_k] for tokens assigned to this expert
            expert_mask = (router_indices == expert_idx)
            
            if expert_mask.any():
                # Count how many tokens use this expert
                expert_counts[expert_idx] = expert_mask.sum().item()
                
                # Get indices of tokens routed to this expert
                # Flatten the mask to [batch_size * seq_len, top_k]
                flat_mask = expert_mask.reshape(-1, self.top_k)
                
                # Get indices of tokens that use this expert
                token_indices = flat_mask.any(dim=-1).nonzero().squeeze(-1)
                
                # Get the routing probabilities for these tokens to this expert
                # Find positions where this expert appears in top-k
                prob_positions = expert_mask.nonzero()
                batch_indices, seq_indices, k_indices = prob_positions[:, 0], prob_positions[:, 1], prob_positions[:, 2]
                token_probs = router_probs_normalized[batch_indices, seq_indices, k_indices]
                
                # Get inputs for this expert
                expert_inputs = flat_input[token_indices]
                
                # Forward through the expert
                expert_output = self.expert_pool.forward_expert(expert_idx, expert_inputs)
                
                # Scale outputs by routing probabilities
                scaled_expert_output = expert_output * token_probs.unsqueeze(-1)
                
                # Combine outputs at the appropriate positions
                # Use index_add_ to handle cases where tokens are routed to same expert multiple times
                expert_outputs.index_add_(0, token_indices, scaled_expert_output)
        
        # Update expert utilization metrics
        self.expert_metrics["expert_utilization"] = expert_counts / (batch_size * seq_len * self.top_k)
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        combined_output = expert_outputs.reshape(batch_size, seq_len, hidden_size)
        
        # Apply dropout and layer norm with residual connection
        combined_output = self.dropout(combined_output)
        final_output = self.layer_norm(combined_output + attention_output)
        
        return final_output
        

class BertMoEEncoder(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Store MoE specific parameters
        self.num_experts = config.num_experts  # Default to 8 if not specified
        self.top_k = config.top_k # Default to 2 if not specified
        
        # Get MoE layer indices from config
        # Can be a list of layer indices [0, 2, 4, 6] or "all" for all layers
        self.moe_layers = getattr(config, 'moe_layers', 'all')
        
        # Convert moe_layers to a set of indices for faster lookup
        if self.moe_layers == 'all':
            self.moe_layer_indices = set(range(config.num_hidden_layers))
        elif isinstance(self.moe_layers, (list, tuple)):
            self.moe_layer_indices = set(self.moe_layers)
        else:
            raise ValueError(f"Invalid moe_layers format: {self.moe_layers}. Must be 'all' or a list of layer indices.")
        
        # Validate layer indices
        for idx in self.moe_layer_indices:
            if not isinstance(idx, int) or idx < 0 or idx >= config.num_hidden_layers:
                raise ValueError(f"Invalid layer index {idx}. Must be an integer in range [0, {config.num_hidden_layers})")
        
        # Create ModuleList with mixed BertMoELayer and BertLayer instances
        self.layer = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if i in self.moe_layer_indices:
                # Use MoE layer
                self.layer.append(BertMoELayer(config, self.num_experts, self.top_k))
            else:
                # Use standard BERT layer
                self.layer.append(BertLayer(config))
        
        self.gradient_checkpointing = False
        
        # Optional: Add metrics tracking for MoE (expert utilization, load balance)
        self.track_expert_metrics = config.track_expert_metrics
        
        if self.track_expert_metrics:
            # Initialize metrics storage (will be updated during forward pass)
            self.expert_metrics = {
                "expert_utilization": torch.zeros(self.num_experts),
                "expert_load_balance": 0.0,
                "moe_layers_used": list(self.moe_layer_indices),
            }
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        
		# For load balancing and tracking (modify 1)
        if self.track_expert_metrics:
            # Reset expert utilization counters
            self.expert_metrics["expert_utilization"] = torch.zeros(self.num_experts, 
                                                                   device=hidden_states.device)
            # Count number of MoE layers actually used
            moe_layers_count = 0
            
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        
        # Process each layer with MoE routing
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # Forward through the MoE layer
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
                
                # If tracking expert metrics and this is a MoE layer, update metrics
                if self.track_expert_metrics and i in self.moe_layer_indices:
                    moe_layers_count += 1
                    if hasattr(layer_module, "expert_metrics"):
                        self.expert_metrics["expert_utilization"] += layer_module.expert_metrics.get(
                            "expert_utilization", torch.zeros(self.num_experts, device=hidden_states.device)
                        )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Final hidden states after all layers
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Calculate final expert utilization metrics across all MoE layers
        if self.track_expert_metrics and moe_layers_count > 0:
            # Normalize by number of MoE layers for average utilization
            self.expert_metrics["expert_utilization"] /= moe_layers_count
            
            # Calculate load balance as coefficient of variation (standard deviation / mean)
            utilization = self.expert_metrics["expert_utilization"]
            if torch.mean(utilization) > 0:
                self.expert_metrics["expert_load_balance"] = torch.std(utilization) / torch.mean(utilization)
            else:
                self.expert_metrics["expert_load_balance"] = torch.tensor(0.0, device=hidden_states.device)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertMoEModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertMoEEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Feed through the MoE encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    
class BertMoEForSequenceClassification(PreTrainedModel):
    """
    BertMoE model with a sequence classification head on top.
    
    This model is designed for classification tasks such as sentiment analysis,
    textual entailment, and text categorization.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # Load the BertMoE model as the base
        self.bert_moe = BertMoEModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(config.classifier_dropout or config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass for sequence classification with BertMoE.
        
        Args:
            input_ids: Token IDs to be processed.
            attention_mask: Mask to avoid attention on padding tokens.
            token_type_ids: Segment token indices to indicate first and second portions of the inputs.
            position_ids: Indices of positions of each input sequence tokens.
            head_mask: Mask to nullify selected heads of the self-attention modules.
            inputs_embeds: Input embeddings instead of input_ids.
            labels: True labels for computing the loss.
            output_attentions: Whether to return attentions weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a dictionary instead of a tuple.
            
        Returns:
            SequenceClassifierOutput or tuple with loss, logits, hidden_states, attentions.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward pass through the BertMoE base model
        outputs = self.bert_moe(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the pooled output for classification
        pooled_output = outputs[1]
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
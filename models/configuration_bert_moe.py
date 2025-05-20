from transformers.configuration_utils import PretrainedConfig

class BertMoEConfig(PretrainedConfig):
    """
    Configuration class for BertMoE model.
    
    This extends the original BertConfig with parameters specific to the
    Mixture of Experts architecture.
    """
    
    model_type = "bert_moe"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        # MoE specific parameters
        num_experts=8,
        top_k=2,
        expert_dropout=0.0,
        expert_init_strategy="identical",  # or "diverse"
        router_temperature=1.0,
        router_noise_epsilon=1e-2,
        router_training_noise=True,
        use_load_balancing=False,
        router_z_loss_coef=1e-3,
        router_aux_loss_coef=0.01,
        track_expert_metrics=True,
        parallel_expert_computation=True,
        **kwargs
    ):
        """
        Initialize BertMoEConfig with MoE-specific parameters.
        
        Args:
            vocab_size: Vocabulary size of the BERT model.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer.
            intermediate_size: Size of the "intermediate" (feed-forward) layer.
            hidden_act: The non-linear activation function in the encoder and pooler.
            hidden_dropout_prob: Dropout probability for all fully connected layers.
            attention_probs_dropout_prob: Dropout ratio for attention probabilities.
            max_position_embeddings: The maximum sequence length for positional embeddings.
            type_vocab_size: The vocabulary size of the `token_type_ids`.
            initializer_range: The standard deviation of the truncated_normal_initializer.
            layer_norm_eps: The epsilon used by LayerNorm.
            pad_token_id: The value used to pad input_ids.
            position_embedding_type: Type of position embedding.
            use_cache: Whether or not the model should return the last key/values attentions.
            classifier_dropout: Dropout probability for classifier.
            
            # MoE specific parameters
            num_experts: Number of expert feed-forward networks per layer.
            top_k: Number of experts to route each token to.
            expert_dropout: Probability of dropping out entire experts during training.
            expert_init_strategy: Strategy for initializing experts ("identical" or "diverse").
            router_temperature: Temperature for router softmax to control sharpness.
            router_noise_epsilon: Magnitude of noise to add to router logits during training.
            router_training_noise: Whether to add noise to router logits during training.
            use_load_balancing: Whether to use auxiliary load balancing loss.
            router_z_loss_coef: Coefficient for router z-loss to improve stability.
            router_aux_loss_coef: Coefficient for router auxiliary losses.
            track_expert_metrics: Whether to track and log expert utilization metrics.
            parallel_expert_computation: Whether to compute expert outputs in parallel.
        """
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            classifier_dropout=classifier_dropout,
            **kwargs
        )
        
        # Store MoE specific parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout = expert_dropout
        self.expert_init_strategy = expert_init_strategy
        self.router_temperature = router_temperature
        self.router_noise_epsilon = router_noise_epsilon
        self.router_training_noise = router_training_noise
        self.use_load_balancing = use_load_balancing
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        self.track_expert_metrics = track_expert_metrics
        self.parallel_expert_computation = parallel_expert_computation
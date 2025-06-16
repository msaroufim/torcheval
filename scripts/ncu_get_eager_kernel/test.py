import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import math

# Optimized Expert implementation
class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process all tokens in a single vectorized operation
        gate = self.act_fn(self.W_gate(x))
        up = self.W_up(x)
        
        # Element-wise multiplication and final projection
        return self.W_down(gate * up)


# Optimized MoEGate implementation
class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Single vectorized operation to compute all routing logits
        logits = self.W_g(x)
        
        # Compute softmax for all tokens at once
        scores = F.softmax(logits, dim=-1)
        
        # Get top-k experts and scores for all tokens in a single operation
        topk_scores, topk_indices = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )

        return topk_indices, topk_scores


# Optimized MoE implementation with efficient expert routing
class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            Expert(config)
            for _ in range(config["n_routed_experts"])
        ])
        self.gating_network = MoEGate(config)

        # Shortcut to computing the sum of shared experts
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute shared expert outputs
        shared_output = self.shared_expert(x)
        
        # Compute expert routing weights
        expert_indices, expert_scores = self.gating_network(x)
        
        # Get shape information
        batch_size, seq_len, hidden_dim = x.shape
        orig_shape = x.shape
        
        # Flatten the input and routing information
        x_flat = x.reshape(-1, hidden_dim)
        expert_indices_flat = expert_indices.reshape(-1)
        expert_scores_flat = expert_scores.reshape(-1)
        
        # Use the optimized inference approach
        routed_output_flat = self.moe_infer(x_flat, expert_indices_flat, expert_scores_flat.unsqueeze(-1))
        
        # Reshape back to original dimensions
        routed_output = routed_output_flat.view(*orig_shape)
        
        # Combine with shared expert output
        return routed_output + shared_output

    @torch.no_grad()
    def moe_infer(self,
                  x: torch.Tensor,
                  flat_expert_indices: torch.Tensor,
                  flat_expert_weights: torch.Tensor
                 ) -> torch.Tensor:
        """
        Efficient inference for MoE by processing tokens per expert
        
        Args:
            x: Flattened input tensor [batch_size*seq_len, hidden_dim]
            flat_expert_indices: Flattened expert indices [batch_size*seq_len*n_experts_per_token]
            flat_expert_weights: Flattened expert weights [batch_size*seq_len*n_experts_per_token, 1]
            
        Returns:
            Processed output tensor [batch_size*seq_len, hidden_dim]
        """
        # Initialize output cache
        expert_cache = torch.zeros_like(x)
        
        # Sort by expert indices to process tokens per expert
        idxs = flat_expert_indices.argsort()
        
        # Count tokens per expert
        counts = flat_expert_indices.bincount(minlength=self.config["n_routed_experts"]).cpu().numpy()
        tokens_per_expert = counts.cumsum()
        
        # Number of experts per token
        num_per_tok = self.config["n_experts_per_token"]
        
        # Get original token indices
        token_idxs = idxs // num_per_tok
        
        # Process each expert
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            
            # Skip if no tokens for this expert
            if start_idx == end_idx:
                continue
            
            # Get the expert
            expert = self.experts[expert_id]
            
            # Get tokens for this expert
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            
            # Process tokens with this expert
            expert_out = expert(expert_tokens)
            
            # Apply weights
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # Add to output using scatter_reduce
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce='sum'
            )
        
        return expert_cache


# Optimized kernel function with proper synchronization
def ref_kernel(data):
    """
    Reference implementation of DeepSeek-style Mixture of Experts using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_dim]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
            
    Returns:
        output: Processed tensor [batch_size, seq_len, d_model]
    """
    input_tensor, weights, config = data
    num_experts = config["n_routed_experts"]
    moe = MoE(config)

    # Time model loading with proper sync
    import time
    torch.cuda.synchronize()
    load_start = time.time()

    # Fill in the given weights of the model using torch.no_grad()
    with torch.no_grad():
        moe.gating_network.W_g.weight = nn.Parameter(weights['router.weight'])

        for i in range(num_experts):
            gate_proj_weight = weights[f'experts.{i}.0.weight']
            up_proj_weight = weights[f'experts.{i}.1.weight']
            down_proj_weight = weights[f'experts.{i}.2.weight']

            # Transpose weights to match expected shape for nn.Linear
            moe.experts[i].W_gate.weight = nn.Parameter(gate_proj_weight.t())
            moe.experts[i].W_up.weight = nn.Parameter(up_proj_weight.t())
            moe.experts[i].W_down.weight = nn.Parameter(down_proj_weight.t())

        moe.shared_expert.W_gate.weight = nn.Parameter(weights['shared_experts.0.weight'].t())
        moe.shared_expert.W_up.weight = nn.Parameter(weights['shared_experts.1.weight'].t())
        moe.shared_expert.W_down.weight = nn.Parameter(weights['shared_experts.2.weight'].t())

    torch.cuda.synchronize()
    load_time = time.time() - load_start

    # Time model execution with proper sync
    torch.cuda.synchronize()
    run_start = time.time()
    
    # Use no_grad for inference
    with torch.no_grad():
        output = moe(input_tensor)
    
    torch.cuda.synchronize()
    run_time = time.time() - run_start

    print(f"Model load time: {load_time:.3f}s")
    print(f"Model run time: {run_time:.3f}s")

    return output


# Input generation function with proper CUDA synchronization
def generate_input(
    dhidden: int,
    dexpert: int,
    nroutedexperts: int,
    nsharedexperts: int,
    nexpertspertoken: int,
    bs: int,
    seqlen: int,
    seed: int
):
    # Configuration dictionary
    config = {
        "d_hidden": dhidden,
        "d_expert": dexpert,
        "n_routed_experts": nroutedexperts,
        "n_shared_experts": nsharedexperts,
        "n_experts_per_token": nexpertspertoken,
        "batch_size": bs,
        "seq_len": seqlen,
        "seed": seed,
    }
    
    # Set seed for reproducibility
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    # Use torch.no_grad() for input generation
    with torch.no_grad():
        # Input tensor
        input_tensor = torch.randn(
            (bs, seqlen, dhidden),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ).contiguous()

        # Initialize router weights
        weights = {}
        weights['router.weight'] = torch.randn(
            (nroutedexperts, dhidden),
            device="cuda",
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(dhidden)

        for i in range(nroutedexperts):
            weights[f'experts.{i}.0.weight'] = torch.randn(
                (dhidden, dexpert),
                device='cuda',
                dtype=torch.float16,
                generator=gen
            ) / math.sqrt(dexpert)

            weights[f'experts.{i}.1.weight'] = torch.randn(
                (dhidden, dexpert),
                device='cuda',
                dtype=torch.float16,
                generator=gen
            ) / math.sqrt(dexpert)

            weights[f'experts.{i}.2.weight'] = torch.randn(
                (dexpert, dhidden),
                device='cuda',
                dtype=torch.float16,
                generator=gen
            ) / math.sqrt(dhidden)
        
        # Shared expert weights
        shared_dim = dexpert * nsharedexperts
        weights['shared_experts.0.weight'] = torch.randn(
            (dhidden, shared_dim),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(shared_dim)
        weights['shared_experts.1.weight'] = torch.randn(
            (dhidden, shared_dim),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(shared_dim)
        weights['shared_experts.2.weight'] = torch.randn(
            (shared_dim, dhidden),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(dhidden)
    
    return (input_tensor, weights, config)


def main():
    # List of configurations to test
    configs = [
        {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 32, "nsharedexperts": 1, "nexpertspertoken": 4, "bs": 1, "seqlen": 512, "seed": 9371},
        {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 32, "nsharedexperts": 1, "nexpertspertoken": 4, "bs": 4, "seqlen": 1024, "seed": 4582},
        {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 32, "nsharedexperts": 1, "nexpertspertoken": 4, "bs": 2, "seqlen": 4096, "seed": 1092},
        {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 32, "nsharedexperts": 1, "nexpertspertoken": 4, "bs": 1, "seqlen": 8192, "seed": 8157},
        {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 32, "nsharedexperts": 1, "nexpertspertoken": 4, "bs": 2, "seqlen": 8192, "seed": 1902},
        # Add the configuration from your colleague
        {"dhidden": 2048, "dexpert": 1408, "nroutedexperts": 64, "nsharedexperts": 1, "nexpertspertoken": 6, "bs": 1, "seqlen": 1024, "seed": 42},
    ]

    for config in configs:
        print(f"\nTesting configuration:")
        print(f"Batch size: {config['bs']}, Sequence length: {config['seqlen']}")
        
        # Clear CUDA cache before each run
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Generate input data
        import time
        start_time = time.time()
        input_tensor, weights, config_dict = generate_input(
            dhidden=config["dhidden"],
            dexpert=config["dexpert"], 
            nroutedexperts=config["nroutedexperts"],
            nsharedexperts=config["nsharedexperts"],
            nexpertspertoken=config["nexpertspertoken"],
            bs=config["bs"],
            seqlen=config["seqlen"],
            seed=config["seed"]
        )
        torch.cuda.synchronize()
        gen_time = time.time() - start_time

        # Run reference kernel
        torch.cuda.synchronize()
        start_time = time.time()
        output = ref_kernel((input_tensor, weights, config_dict))
        torch.cuda.synchronize()
        run_time = time.time() - start_time

        print(f"Generate time: {gen_time:.3f}s")
        print(f"Run time: {run_time:.3f}s")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()

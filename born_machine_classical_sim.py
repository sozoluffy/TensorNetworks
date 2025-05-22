# born_machine_classical_sim.py
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from utils import generate_all_binary_outcomes, get_binary_key

class ClassicalBornMachine(nn.Module):
    """
    A classical simulation of a Born Machine for discrete binary variables.
    It directly parameterizes the probabilities of each outcome, or uses logits.
    This is a simplified stand-in for an actual quantum circuit.

    The "parameters" of this Born machine will be directly optimized.
    For amortization (conditioning on x), the input x can modify the parameters or be
    part of the input to a neural network that generates the distribution parameters.
    """
    def __init__(self, num_latent_vars, use_logits=True, conditioning_dim=0):
        """
        Args:
            num_latent_vars (int): The number of discrete binary latent variables (qubits in quantum analogy).
            use_logits (bool): If True, parameters are logits and softmax is applied.
                               If False, parameters are directly probabilities (must be kept positive and sum to 1).
            conditioning_dim (int): Dimension of the conditioning vector x. If 0, no conditioning.
                                    If > 0, a simple linear layer will be used to incorporate x.
        """
        super().__init__()
        self.num_latent_vars = num_latent_vars
        self.num_outcomes = 2**num_latent_vars
        self.use_logits = use_logits
        self.conditioning_dim = conditioning_dim

        if self.conditioning_dim > 0:
            # A simple network to process conditioning variable x and produce logits/probabilities
            # This is a basic form of amortization.
            # The output size is self.num_outcomes
            self.param_generator_net = nn.Sequential(
                nn.Linear(self.conditioning_dim, 64), # Example hidden layer size
                nn.ReLU(),
                nn.Linear(64, self.num_outcomes)
            )
        else:
            # Direct parameters for the distribution (logits or probabilities)
            if self.use_logits:
                # Initialize logits (e.g., to zeros for a uniform initial distribution after softmax)
                self.params = nn.Parameter(torch.zeros(self.num_outcomes))
            else:
                # Initialize probabilities (e.g., uniform)
                # These must be constrained during optimization (e.g., projected onto simplex)
                # Or, use a transformation like softmax implicitly. For simplicity with PyTorch,
                # logits are generally easier to manage.
                initial_probs = torch.ones(self.num_outcomes) / self.num_outcomes
                self.params = nn.Parameter(initial_probs)


        # Store all possible outcome tuples in the order corresponding to parameter indices
        self.all_outcome_tuples = generate_all_binary_outcomes(self.num_latent_vars)


    def get_probabilities(self, x_condition=None):
        """
        Returns the probability distribution q_theta(z) or q_theta(z|x).

        Args:
            x_condition (torch.Tensor, optional): The conditioning variable.
                Shape (batch_size, conditioning_dim) or (conditioning_dim,).
                If None and conditioning_dim > 0, an error is raised.

        Returns:
            torch.Tensor: A 1D tensor of probabilities for each outcome, summing to 1.
                          Shape (num_outcomes,) or (batch_size, num_outcomes) if x_condition is batched.
        """
        if self.conditioning_dim > 0:
            if x_condition is None:
                raise ValueError("x_condition must be provided for a conditional Born machine.")
            if x_condition.ndim == 1:
                x_condition = x_condition.unsqueeze(0) # Add batch dimension if not present
            
            raw_params = self.param_generator_net(x_condition) # Shape (batch_size, num_outcomes)
        else:
            if x_condition is not None:
                # Or, one could choose to ignore x_condition if conditioning_dim is 0.
                # For clarity, let's be strict.
                raise ValueError("x_condition was provided, but conditioning_dim is 0.")
            raw_params = self.params.unsqueeze(0) # Add batch dimension, shape (1, num_outcomes)

        if self.use_logits:
            # Apply softmax to convert logits to probabilities
            # Logits are unconstrained, softmax ensures positive and sum to 1.
            return torch.softmax(raw_params, dim=-1) # Apply softmax along the last dimension
        else:
            # If params are direct probabilities, they should already be valid.
            # This mode is harder to optimize correctly without constraints.
            # For safety, one might re-normalize, but ideally, the optimization handles it.
            # Or, ensure params are passed through a softmax-like layer if not using logits.
            # For now, assume they are valid if not using logits.
            # This branch is less recommended with gradient descent unless parameters are carefully handled.
            if not torch.all(raw_params >= 0) or not torch.allclose(torch.sum(raw_params, dim=-1), torch.tensor(1.0)):
                 # This is a check, but direct optimization of probabilities is tricky.
                 # Consider applying a final softmax even if 'use_logits' was initially false,
                 # or use a different parameterization if direct probabilities are desired.
                 # For this project, sticking to logits is safer.
                 print("Warning: Direct probabilities are not valid. Consider using logits or ensuring constraints.")
                 return torch.softmax(raw_params, dim=-1) # Fallback to softmax for safety
            return raw_params

    def sample(self, num_samples=1, x_condition=None):
        """
        Samples from the Born machine's distribution q_theta(z) or q_theta(z|x).

        Args:
            num_samples (int): The number of samples to generate.
            x_condition (torch.Tensor, optional): The conditioning variable.
                If provided, must match conditioning_dim.
                If x_condition has a batch dimension, num_samples will be drawn for *each* item in the batch.

        Returns:
            torch.Tensor: A tensor of shape (num_samples, num_latent_vars) containing binary samples (0 or 1).
                          If x_condition is batched (B, D), output is (B, num_samples, num_latent_vars).
        """
        probs = self.get_probabilities(x_condition) # Shape (1, num_outcomes) or (B, num_outcomes)
        
        if probs.ndim == 2 and probs.shape[0] > 1: # Batched x_condition
            batch_size = probs.shape[0]
            samples_list = []
            for i in range(batch_size):
                # For each item in the batch, sample num_samples times
                # multinomial expects 1D probabilities for each draw.
                # It returns indices of the chosen outcomes.
                sampled_indices = torch.multinomial(probs[i], num_samples, replacement=True) # Shape (num_samples,)
                # Convert indices to actual outcome tuples/tensors
                batch_samples = torch.tensor([self.all_outcome_tuples[idx.item()] for idx in sampled_indices], dtype=torch.float32)
                samples_list.append(batch_samples)
            return torch.stack(samples_list) # Shape (B, num_samples, num_latent_vars)
        else: # Unbatched or conditioning_dim = 0
            probs_1d = probs.squeeze(0) # Shape (num_outcomes,)
            sampled_indices = torch.multinomial(probs_1d, num_samples, replacement=True) # Shape (num_samples,)
            samples = torch.tensor([self.all_outcome_tuples[idx.item()] for idx in sampled_indices], dtype=torch.float32)
            return samples # Shape (num_samples, num_latent_vars)


    def get_prob_dict(self, x_condition=None):
        """
        Returns the probability distribution as a dictionary: {outcome_tuple: probability}.
        Useful for comparison with BayesianNetwork's posterior format.

        Args:
            x_condition (torch.Tensor, optional): The conditioning variable.

        Returns:
            dict: Probability distribution.
        """
        probs_tensor = self.get_probabilities(x_condition)
        if probs_tensor.shape[0] > 1:
            # This function is intended for a single distribution, not a batch.
            # User should call this for a specific x_condition instance if batched.
            raise ValueError("get_prob_dict is for a single distribution. Provide a specific x_condition or handle batching outside.")
        
        probs_1d = probs_tensor.squeeze(0).detach().cpu().numpy()
        
        prob_dict = {}
        for i, outcome_tuple in enumerate(self.all_outcome_tuples):
            prob_dict[outcome_tuple] = probs_1d[i]
        return prob_dict

    def get_log_q_z_x(self, z_samples, x_condition=None):
        """
        Calculates log q_theta(z | x) for given samples z and condition x.
        This is needed for the VAE-like objective in adversarial VI.

        Args:
            z_samples (torch.Tensor): Samples from the latent space, shape (batch_size, num_latent_vars).
            x_condition (torch.Tensor, optional): Conditioning variables, shape (batch_size, conditioning_dim).
                                                 If conditioning_dim is 0, this can be None.

        Returns:
            torch.Tensor: log probabilities, shape (batch_size,).
        """
        if self.conditioning_dim > 0 and x_condition is None:
            raise ValueError("x_condition must be provided for a conditional Born machine.")
        if self.conditioning_dim == 0 and x_condition is not None:
             # Or, allow it and ignore x_condition. For clarity, let's be strict.
            raise ValueError("x_condition provided but Born machine is not conditional.")
        if self.conditioning_dim > 0 and x_condition.shape[0] != z_samples.shape[0]:
            raise ValueError("Batch size of x_condition and z_samples must match.")

        # Get the full probability distribution(s) q(Z|x) or q(Z)
        # probs will have shape (batch_size, num_outcomes) or (1, num_outcomes)
        probs = self.get_probabilities(x_condition)
        
        if self.conditioning_dim == 0 and z_samples.shape[0] > 1 and probs.shape[0] == 1:
            # If not conditional, probs is (1, num_outcomes).
            # We need to match it to the batch size of z_samples.
            probs = probs.repeat(z_samples.shape[0], 1) # Repeat for each sample in z_batch

        log_probs = torch.log(probs + 1e-9) # Add epsilon for numerical stability

        # For each sample in z_samples, find its corresponding log probability
        selected_log_probs = []
        for i in range(z_samples.shape[0]):
            sample_tuple = tuple(z_samples[i].cpu().long().tolist()) # Convert tensor row to tuple of ints
            try:
                outcome_idx = self.all_outcome_tuples.index(sample_tuple)
                selected_log_probs.append(log_probs[i, outcome_idx])
            except ValueError:
                # This should not happen if z_samples are valid outcomes
                raise ValueError(f"Sample {sample_tuple} is not a valid outcome for this Born machine.")
        
        return torch.stack(selected_log_probs)


if __name__ == '__main__':
    # --- Test Unconditional Born Machine ---
    print("--- Unconditional Born Machine (2 latent vars) ---")
    bm_uncond = ClassicalBornMachine(num_latent_vars=2, use_logits=True)
    print("Initial parameters (logits):", bm_uncond.params)
    
    initial_probs_uncond = bm_uncond.get_probabilities()
    print("Initial probabilities q(z):", initial_probs_uncond)
    print("Sum of probs:", torch.sum(initial_probs_uncond))

    samples_uncond = bm_uncond.sample(num_samples=5)
    print("Samples from q(z):\n", samples_uncond)

    prob_dict_uncond = bm_uncond.get_prob_dict()
    print("Prob dict q(z):", prob_dict_uncond)

    # Test get_log_q_z_x for unconditional
    test_z_samples = torch.tensor([[0,0], [0,1], [1,1]], dtype=torch.float32)
    log_q_z = bm_uncond.get_log_q_z_x(test_z_samples)
    print(f"Log q(z) for samples {test_z_samples.tolist()}: {log_q_z.exp()}") # Print exp to check with probs

    # --- Test Conditional Born Machine ---
    print("\n--- Conditional Born Machine (2 latent vars, cond_dim=3) ---")
    cond_dim = 3
    bm_cond = ClassicalBornMachine(num_latent_vars=2, use_logits=True, conditioning_dim=cond_dim)
    
    # Example conditioning vectors
    x_cond1 = torch.randn(cond_dim) 
    x_cond2 = torch.randn(cond_dim)
    x_batch = torch.stack([x_cond1, x_cond2]) # Batch of 2

    print(f"Shape of x_batch: {x_batch.shape}")

    probs_cond_batch = bm_cond.get_probabilities(x_condition=x_batch)
    print("Probabilities q(z|x) for batch:\n", probs_cond_batch)
    print("Sum of probs for each x in batch:", torch.sum(probs_cond_batch, dim=1))

    samples_cond_batch = bm_cond.sample(num_samples=3, x_condition=x_batch)
    print(f"Samples from q(z|x) for batch (shape {samples_cond_batch.shape}):\n", samples_cond_batch)
    # samples_cond_batch will have shape (batch_size, num_samples, num_latent_vars)
    # e.g., (2, 3, 2)

    # Get prob dict for a single conditioning input
    probs_cond_single_x = bm_cond.get_probabilities(x_condition=x_cond1)
    prob_dict_cond_single_x = bm_cond.get_prob_dict(x_condition=x_cond1)
    print(f"Prob dict q(z|x1) for x1: {prob_dict_cond_single_x}")

    # Test get_log_q_z_x for conditional
    # Let's use the samples generated for the batch, but we need to be careful with shapes
    # z_samples should be (B, num_latent_vars)
    # x_condition should be (B, cond_dim)
    
    # Example: take the first sample for each batch item
    z_samples_for_log_q = samples_cond_batch[:, 0, :] # Shape (B, num_latent_vars)
    log_q_z_given_x = bm_cond.get_log_q_z_x(z_samples_for_log_q, x_condition=x_batch)
    print(f"Log q(z|x) for z_samples_for_log_q and x_batch (exp values): {log_q_z_given_x.exp()}")

    # Check consistency:
    for i in range(x_batch.shape[0]):
        z_val_tuple = tuple(z_samples_for_log_q[i].long().tolist())
        manual_prob = bm_cond.get_prob_dict(x_condition=x_batch[i])[z_val_tuple]
        print(f"  Batch item {i}: Sampled z={z_val_tuple}, log_q_exp={log_q_z_given_x.exp()[i]:.4f}, direct_prob={manual_prob:.4f}")


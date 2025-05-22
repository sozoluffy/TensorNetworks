# born_machine_classical_sim.py
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax # This import seems unused, consider removing if not needed elsewhere
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
            self.param_generator_net = nn.Sequential(
                nn.Linear(self.conditioning_dim, max(self.conditioning_dim * 2, 32)), # Ensure hidden layer isn't too small
                nn.ReLU(),
                nn.Linear(max(self.conditioning_dim * 2, 32), self.num_outcomes)
            )
        else:
            if self.use_logits:
                self.params = nn.Parameter(torch.zeros(self.num_outcomes))
            else:
                initial_probs = torch.ones(self.num_outcomes) / self.num_outcomes
                self.params = nn.Parameter(initial_probs)

        self.all_outcome_tuples = generate_all_binary_outcomes(self.num_latent_vars)


    def get_probabilities(self, x_condition=None):
        """
        Returns the probability distribution q_theta(z) or q_theta(z|x).

        Args:
            x_condition (torch.Tensor, optional): The conditioning variable.
                Shape (batch_size, conditioning_dim) or (conditioning_dim,).
                If None and conditioning_dim > 0, an error is raised.

        Returns:
            torch.Tensor: A tensor of probabilities.
                          Shape (num_outcomes,) if x_condition is None or a single instance.
                          Shape (batch_size, num_outcomes) if x_condition is batched.
        """
        if self.conditioning_dim > 0:
            if x_condition is None:
                raise ValueError("x_condition must be provided for a conditional Born machine.")
            
            # Add batch dimension if x_condition is a single instance
            if x_condition.ndim == 1:
                x_condition_batched = x_condition.unsqueeze(0) # Shape (1, conditioning_dim)
            else:
                x_condition_batched = x_condition # Shape (batch_size, conditioning_dim)
            
            raw_params = self.param_generator_net(x_condition_batched) # Shape (batch_size_x, num_outcomes)
        else:
            if x_condition is not None:
                raise ValueError("x_condition was provided, but conditioning_dim is 0.")
            # For unconditioned, output shape (1, num_outcomes) to be consistent
            raw_params = self.params.unsqueeze(0) 

        if self.use_logits:
            return torch.softmax(raw_params, dim=-1)
        else:
            # This branch is less recommended due to optimization difficulties
            # Forcing softmax for safety, assuming params might not be valid probabilities
            print("Warning: Direct probabilities are used; ensuring softmax for validity.")
            return torch.softmax(raw_params, dim=-1)


    def sample(self, num_samples=1, x_condition=None):
        """
        Samples from the Born machine's distribution q_theta(z) or q_theta(z|x).

        Args:
            num_samples (int): The number of samples to generate per item in the (potential) x_condition batch.
            x_condition (torch.Tensor, optional): The conditioning variable.
                If shape (D,), it's treated as a single condition.
                If shape (B, D), it's treated as a batch of B conditions.

        Returns:
            torch.Tensor:
                If x_condition is None or single: (num_samples, num_latent_vars)
                If x_condition is batched (B, D): (B, num_samples, num_latent_vars)
        """
        probs = self.get_probabilities(x_condition) # Shape (B_x, num_outcomes) or (1, num_outcomes)
        
        is_batched_condition = probs.shape[0] > 1 or (x_condition is not None and x_condition.ndim > 1)

        if is_batched_condition:
            batch_size_x = probs.shape[0]
            samples_list = []
            for i in range(batch_size_x):
                sampled_indices = torch.multinomial(probs[i], num_samples, replacement=True)
                batch_samples = torch.tensor([self.all_outcome_tuples[idx.item()] for idx in sampled_indices], dtype=torch.float32, device=probs.device)
                samples_list.append(batch_samples)
            return torch.stack(samples_list)
        else: # Single condition or unconditioned
            probs_1d = probs.squeeze(0)
            sampled_indices = torch.multinomial(probs_1d, num_samples, replacement=True)
            samples = torch.tensor([self.all_outcome_tuples[idx.item()] for idx in sampled_indices], dtype=torch.float32, device=probs.device)
            return samples


    def get_prob_dict(self, x_condition=None):
        """
        Returns the probability distribution as a dictionary: {outcome_tuple: probability}.
        Intended for a single x_condition or unconditioned.
        """
        probs_tensor = self.get_probabilities(x_condition) # Will be shape (1, num_outcomes) or (num_outcomes) if unconditioned & squeezed
        
        if probs_tensor.shape[0] > 1 and not (probs_tensor.ndim == 1 and self.conditioning_dim==0) : # Check if it's a batch of distributions
            raise ValueError("get_prob_dict is for a single distribution. Provide a specific single x_condition or handle batching outside.")
        
        probs_1d = probs_tensor.squeeze().detach().cpu().numpy() # Squeeze to 1D
        
        prob_dict = {}
        for i, outcome_tuple in enumerate(self.all_outcome_tuples):
            prob_dict[outcome_tuple] = probs_1d[i]
        return prob_dict

    def get_log_q_z_x(self, z_samples, x_condition=None):
        """
        Calculates log q_theta(z | x) for given samples z and condition x.

        Args:
            z_samples (torch.Tensor): Samples from the latent space.
                                      Shape (batch_size_z, num_latent_vars).
            x_condition (torch.Tensor, optional): Conditioning variables.
                                                 Shape (conditioning_dim,) for a single condition,
                                                 or (batch_size_x, conditioning_dim) for batched conditions.
                                                 If batch_size_x = 1, it applies to all z_samples.
                                                 If batch_size_x = batch_size_z, it's element-wise.

        Returns:
            torch.Tensor: log probabilities, shape (batch_size_z,).
        """
        if self.conditioning_dim > 0 and x_condition is None:
            raise ValueError("x_condition must be provided for a conditional Born machine.")
        if self.conditioning_dim == 0 and x_condition is not None:
            raise ValueError("x_condition provided but Born machine is not conditional.")

        # probs will have shape (batch_size_x, num_outcomes) or (1, num_outcomes)
        # where batch_size_x depends on x_condition's shape (is 1 if x_condition is single)
        probs = self.get_probabilities(x_condition) 
        log_probs_distributions = torch.log(probs + 1e-9) # Shape (batch_size_x, num_outcomes)

        batch_size_z = z_samples.shape[0]
        batch_size_x = probs.shape[0]

        selected_log_probs = []

        if batch_size_x == 1:
            # Single conditioning applies to all z_samples
            current_log_probs_dist = log_probs_distributions.squeeze(0) # Shape (num_outcomes,)
            for i in range(batch_size_z):
                sample_tuple = tuple(z_samples[i].cpu().long().tolist())
                try:
                    outcome_idx = self.all_outcome_tuples.index(sample_tuple)
                    selected_log_probs.append(current_log_probs_dist[outcome_idx])
                except ValueError:
                    raise ValueError(f"Sample {sample_tuple} is not a valid outcome for this Born machine.")
        elif batch_size_x == batch_size_z:
            # Batched x_condition matches batched z_samples
            for i in range(batch_size_z):
                sample_tuple = tuple(z_samples[i].cpu().long().tolist())
                try:
                    outcome_idx = self.all_outcome_tuples.index(sample_tuple)
                    selected_log_probs.append(log_probs_distributions[i, outcome_idx])
                except ValueError:
                    raise ValueError(f"Sample {sample_tuple} is not a valid outcome for this Born machine.")
        else:
            raise ValueError(f"Batch size of x_condition ({batch_size_x}) must be 1 or match z_samples batch size ({batch_size_z}).")
        
        return torch.stack(selected_log_probs)
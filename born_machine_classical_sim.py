# born_machine_classical_sim.py
import numpy as np
import torch
import torch.nn as nn
from utils import generate_all_binary_outcomes, get_binary_key

class ClassicalBornMachine(nn.Module):
    """
    Improved classical Born machine with better initialization and architecture options.
    """
    def __init__(self, num_latent_vars, use_logits=True, conditioning_dim=0, 
                 init_method='small_random', hidden_dims=None, use_layer_norm=False):
        super().__init__()
        self.num_latent_vars = num_latent_vars
        self.num_outcomes = 2**num_latent_vars
        self.use_logits = use_logits
        self.conditioning_dim = conditioning_dim
        self.use_layer_norm = use_layer_norm
        
        # Add fixed probability mode
        self._fixed_probs = None
        self._use_fixed_probs = False

        if self.conditioning_dim > 0:
            # Enhanced conditional network
            if hidden_dims is None:
                hidden_dims = [max(self.conditioning_dim * 4, 64), max(self.conditioning_dim * 2, 32)]
            
            layers = []
            current_dim = self.conditioning_dim
            
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                current_dim = h_dim
            
            layers.append(nn.Linear(current_dim, self.num_outcomes))
            
            self.param_generator_net = nn.Sequential(*layers)
            
            # Initialize network weights
            for m in self.param_generator_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                    
        else:
            # Initialize parameters based on method
            if init_method == 'zero':
                self.params = nn.Parameter(torch.zeros(self.num_outcomes))
            elif init_method == 'small_random':
                self.params = nn.Parameter(0.1 * torch.randn(self.num_outcomes))
            elif init_method == 'uniform':
                uniform_logit = torch.log(torch.ones(self.num_outcomes) / self.num_outcomes)
                self.params = nn.Parameter(uniform_logit + 0.01 * torch.randn(self.num_outcomes))
            else:
                self.params = nn.Parameter(torch.randn(self.num_outcomes))

        self.all_outcome_tuples = generate_all_binary_outcomes(self.num_latent_vars)

    def set_fixed_probs(self, prob_tensor):
        """Set fixed probabilities for the Born machine."""
        self._fixed_probs = prob_tensor.detach().clone()
        self._use_fixed_probs = True

    def clear_fixed_probs(self):
        """Clear fixed probabilities and return to normal mode."""
        self._fixed_probs = None
        self._use_fixed_probs = False

    def get_probabilities(self, x_condition=None):
        """Returns probability distribution with improved numerical stability."""
        # If using fixed probabilities, return them
        if self._use_fixed_probs and self._fixed_probs is not None:
            return self._fixed_probs.unsqueeze(0) if self._fixed_probs.ndim == 1 else self._fixed_probs
        
        if self.conditioning_dim > 0:
            if x_condition is None:
                raise ValueError("x_condition must be provided for conditional Born machine.")
            
            if x_condition.ndim == 1:
                x_condition_batched = x_condition.unsqueeze(0)
            else:
                x_condition_batched = x_condition
            
            raw_params = self.param_generator_net(x_condition_batched)
        else:
            if x_condition is not None:
                raise ValueError("x_condition provided but conditioning_dim is 0.")
            raw_params = self.params.unsqueeze(0)

        if self.use_logits:
            return torch.softmax(raw_params - raw_params.max(dim=-1, keepdim=True)[0], dim=-1)
        else:
            probs = torch.abs(raw_params)
            return probs / probs.sum(dim=-1, keepdim=True)

    def sample(self, num_samples=1, x_condition=None):
        """Sample with improved numerical stability."""
        probs = self.get_probabilities(x_condition)
        
        probs = probs + 1e-10
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        is_batched_condition = probs.shape[0] > 1 or (x_condition is not None and x_condition.ndim > 1)

        if is_batched_condition:
            batch_size_x = probs.shape[0]
            samples_list = []
            for i in range(batch_size_x):
                sampled_indices = torch.multinomial(probs[i], num_samples, replacement=True)
                batch_samples = torch.tensor([self.all_outcome_tuples[idx.item()] for idx in sampled_indices], 
                                           dtype=torch.float32, device=probs.device)
                samples_list.append(batch_samples)
            return torch.stack(samples_list)
        else:
            probs_1d = probs.squeeze(0)
            sampled_indices = torch.multinomial(probs_1d, num_samples, replacement=True)
            samples = torch.tensor([self.all_outcome_tuples[idx.item()] for idx in sampled_indices], 
                                 dtype=torch.float32, device=probs.device)
            return samples

    def get_prob_dict(self, x_condition=None):
        """Returns probability distribution as dictionary."""
        probs_tensor = self.get_probabilities(x_condition)
        
        if probs_tensor.shape[0] > 1 and not (probs_tensor.ndim == 1 and self.conditioning_dim == 0):
            raise ValueError("get_prob_dict is for a single distribution.")
        
        probs_1d = probs_tensor.squeeze().detach().cpu().numpy()
        
        prob_dict = {}
        for i, outcome_tuple in enumerate(self.all_outcome_tuples):
            prob_dict[outcome_tuple] = probs_1d[i]
        return prob_dict

    def get_log_q_z_x(self, z_samples, x_condition=None):
        """Calculate log probabilities with improved numerical stability."""
        if self.conditioning_dim > 0 and x_condition is None:
            raise ValueError("x_condition must be provided for conditional Born machine.")
        if self.conditioning_dim == 0 and x_condition is not None:
            raise ValueError("x_condition provided but Born machine is not conditional.")

        probs = self.get_probabilities(x_condition)
        log_probs_distributions = torch.log(probs.clamp(min=1e-10))

        batch_size_z = z_samples.shape[0]
        batch_size_x = probs.shape[0]

        selected_log_probs = []

        if batch_size_x == 1:
            current_log_probs_dist = log_probs_distributions.squeeze(0)
            for i in range(batch_size_z):
                sample_tuple = tuple(z_samples[i].cpu().long().tolist())
                try:
                    outcome_idx = self.all_outcome_tuples.index(sample_tuple)
                    selected_log_probs.append(current_log_probs_dist[outcome_idx])
                except ValueError:
                    raise ValueError(f"Sample {sample_tuple} is not a valid outcome.")
        elif batch_size_x == batch_size_z:
            for i in range(batch_size_z):
                sample_tuple = tuple(z_samples[i].cpu().long().tolist())
                try:
                    outcome_idx = self.all_outcome_tuples.index(sample_tuple)
                    selected_log_probs.append(log_probs_distributions[i, outcome_idx])
                except ValueError:
                    raise ValueError(f"Sample {sample_tuple} is not a valid outcome.")
        else:
            raise ValueError(f"Batch size mismatch: x_condition ({batch_size_x}) vs z_samples ({batch_size_z}).")
        
        return torch.stack(selected_log_probs)

    def entropy(self, x_condition=None):
        """Calculate entropy of the distribution for regularization."""
        probs = self.get_probabilities(x_condition).squeeze()
        log_probs = torch.log(probs.clamp(min=1e-10))
        return -(probs * log_probs).sum()
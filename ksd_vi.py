# ksd_vi.py
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
import numpy as np
from functools import partial
import copy

from bayesian_network import BayesianNetwork
from born_machine_classical_sim import ClassicalBornMachine
from stein_utils import (
    base_hamming_kernel_torch,
    get_score_function_sp_for_z,
    get_stein_kernel_kp_value
)
from utils import calculate_tvd, generate_all_binary_outcomes


class KSDVariationalInference:
    def __init__(self, bayesian_network, latent_vars_names, observed_vars_names,
                 born_machine_config, base_kernel_length_scale=1.0, device='cpu'):
        self.bn = bayesian_network
        self.latent_vars_names = latent_vars_names
        self.observed_vars_names = observed_vars_names
        self.num_latent_vars = len(latent_vars_names)
        self.num_observed_vars = len(observed_vars_names)
        self.device = device

        # Enhanced Born machine configuration
        born_machine_config = {**born_machine_config, 'init_method': 'small_random'}
        self.born_machine = ClassicalBornMachine(num_latent_vars=self.num_latent_vars,
                                                 **born_machine_config).to(device)

        self.all_latent_states_tuples = generate_all_binary_outcomes(self.num_latent_vars)
        self.num_possible_latent_states = len(self.all_latent_states_tuples)

        self.base_kernel_func = partial(base_hamming_kernel_torch,
                                        num_vars=self.num_latent_vars,
                                        length_scale=base_kernel_length_scale)
        
        self._score_function_cache = {}

    def _get_precomputed_s_p(self, z_tuple, x_dict):
        if z_tuple in self._score_function_cache:
            return self._score_function_cache[z_tuple]
        
        s_p_val = get_score_function_sp_for_z(
            self.bn, x_dict, z_tuple,
            self.latent_vars_names, self.observed_vars_names,
            device=self.device
        )
        self._score_function_cache[z_tuple] = s_p_val
        return s_p_val

    def _precompute_all_s_p(self, x_dict):
        self._score_function_cache.clear()
        print("Precomputing score functions s_p(x,z)...")
        for z_tuple in self.all_latent_states_tuples:
            self._get_precomputed_s_p(z_tuple, x_dict)
        print("Score functions precomputed.")

    def train(self, x_observation_dict, num_epochs, lr_born_machine,
              verbose=True, true_posterior_for_tvd=None,
              use_lr_scheduler=True, gradient_clip_norm=10.0,
              optimizer_type="adam", adam_betas=(0.9, 0.999),
              entropy_weight=0.01, patience=200):
        
        if self.num_observed_vars > 0 and set(x_observation_dict.keys()) != set(self.observed_vars_names):
            raise ValueError("Keys in x_observation_dict must match self.observed_vars_names.")

        x_obs_list = [x_observation_dict[name] for name in self.observed_vars_names] if self.num_observed_vars > 0 else []
        x_obs_tensor_for_bm = torch.tensor(x_obs_list, dtype=torch.float32, device=self.device)
        
        born_machine_x_condition = None
        if self.born_machine.conditioning_dim > 0:
            if self.num_observed_vars == 0:
                raise ValueError("Born machine is conditional but no observed vars specified.")
            if self.born_machine.conditioning_dim != self.num_observed_vars:
                raise ValueError("Born machine conditioning_dim must match num_observed_vars.")
            born_machine_x_condition = x_obs_tensor_for_bm

        self._precompute_all_s_p(x_observation_dict)

        # Choose optimizer
        if optimizer_type == "adam":
            optimizer_born = optim.Adam(self.born_machine.parameters(), lr=lr_born_machine, betas=adam_betas)
        else:
            optimizer_born = optim.SGD(self.born_machine.parameters(), lr=lr_born_machine, momentum=0.9)

        # Learning rate scheduler
        scheduler = None
        if use_lr_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_born, T_max=num_epochs, eta_min=lr_born_machine/10)

        history = {'loss_ksd': [], 'tvd': [], 'grad_norm': [], 'entropy': []}
        
        best_tvd = float('inf')
        best_epoch = -1
        best_probs = None  # Store best probabilities directly
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            optimizer_born.zero_grad()

            q_probs_all_z = self.born_machine.get_probabilities(x_condition=born_machine_x_condition).squeeze()
            
            if q_probs_all_z.shape[0] != self.num_possible_latent_states:
                raise ValueError(f"Probabilities shape mismatch: {q_probs_all_z.shape}")

            # Calculate KSD with improved numerical stability
            sum_k_p_weighted = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            q_probs_all_z_64 = q_probs_all_z.to(torch.float64)

            for i in range(self.num_possible_latent_states):
                zi_tuple = self.all_latent_states_tuples[i]
                qi = q_probs_all_z_64[i]
                spi = self._score_function_cache[zi_tuple].to(torch.float64)

                for j in range(self.num_possible_latent_states):
                    zj_tuple = self.all_latent_states_tuples[j]
                    qj = q_probs_all_z_64[j]
                    spj = self._score_function_cache[zj_tuple].to(torch.float64)

                    kp_val = get_stein_kernel_kp_value(
                        zi_tuple, zj_tuple, x_observation_dict,
                        self.bn, self.latent_vars_names, self.observed_vars_names,
                        self.base_kernel_func,
                        spi, spj,
                        device=self.device
                    ).to(torch.float64)
                    sum_k_p_weighted += qi * qj * kp_val
            
            K_x_theta = sum_k_p_weighted
            ksd_loss = torch.sqrt(K_x_theta.clamp(min=1e-12))
            
            # Add entropy regularization for exploration
            entropy = self.born_machine.entropy(x_condition=born_machine_x_condition)
            
            # Total loss
            loss = ksd_loss - entropy_weight * entropy
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss: {loss.item()}. Skipping update.")
            else:
                loss.backward()
                
                # Gradient clipping
                grad_norm = nn_utils.clip_grad_norm_(self.born_machine.parameters(), gradient_clip_norm)
                
                optimizer_born.step()
                
                if scheduler is not None:
                    scheduler.step()
            
            history['loss_ksd'].append(ksd_loss.item())
            history['grad_norm'].append(grad_norm.item() if 'grad_norm' in locals() else 0.0)
            history['entropy'].append(entropy.item() if 'entropy' in locals() else 0.0)

            # Calculate TVD and save best probabilities
            if true_posterior_for_tvd is not None:
                current_q_dist_dict = self.born_machine.get_prob_dict(x_condition=born_machine_x_condition)
                tvd = calculate_tvd(true_posterior_for_tvd, current_q_dist_dict)
                history['tvd'].append(tvd)
                
                # Save best probabilities directly
                if tvd < best_tvd:
                    best_tvd = tvd
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    
                    # Store the actual probability tensor
                    with torch.no_grad():
                        best_probs = self.born_machine.get_probabilities(x_condition=born_machine_x_condition).squeeze().clone()
                    
                    if verbose and tvd < 0.05:
                        print(f"  -> New best TVD: {tvd:.6f} at epoch {epoch+1}")
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement > patience and epoch > 300:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
            else:
                history['tvd'].append(np.nan)

            if verbose and (epoch % max(1, num_epochs // 20) == 0 or epoch == num_epochs - 1):
                log_msg = f"Epoch {epoch+1}/{num_epochs} | KSD: {ksd_loss.item():.6f}"
                if scheduler is not None:
                    log_msg += f" | LR: {scheduler.get_last_lr()[0]:.6f}"
                if 'entropy' in locals():
                    log_msg += f" | Entropy: {entropy.item():.4f}"
                if true_posterior_for_tvd and not np.isnan(history['tvd'][-1]):
                    log_msg += f" | TVD: {history['tvd'][-1]:.6f}"
                print(log_msg)
        
        # Restore best probabilities
        if best_probs is not None:
            if verbose:
                print(f"\nRestoring best probabilities (TVD: {best_tvd:.6f} from epoch {best_epoch+1})")
            
            # Set the Born machine to use fixed probabilities
            self.born_machine.set_fixed_probs(best_probs)
            
            # Verify
            final_q_dist = self.born_machine.get_prob_dict(x_condition=born_machine_x_condition)
            final_tvd = calculate_tvd(true_posterior_for_tvd, final_q_dist)
            
            if abs(final_tvd - best_tvd) > 1e-6:
                print(f"WARNING: Still have restoration issue! Expected TVD: {best_tvd:.6f}, Got: {final_tvd:.6f}")
            else:
                if verbose:
                    print(f"Successfully restored best probabilities! Final TVD: {final_tvd:.6f}")
        
        return history
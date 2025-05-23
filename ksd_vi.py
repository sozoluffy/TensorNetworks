# ksd_vi.py
import torch
import torch.optim as optim
import numpy as np
from functools import partial

from bayesian_network import BayesianNetwork
from born_machine_classical_sim import ClassicalBornMachine
from stein_utils import (
    base_hamming_kernel_torch, 
    get_score_function_sp_for_z, 
    get_stein_kernel_kp_value
)
from utils import calculate_tvd, generate_all_binary_outcomes # Assuming these are in your main utils.py


class KSDVariationalInference:
    def __init__(self, bayesian_network, latent_vars_names, observed_vars_names,
                 born_machine_config, base_kernel_length_scale=1.0, device='cpu'):
        self.bn = bayesian_network
        self.latent_vars_names = latent_vars_names
        self.observed_vars_names = observed_vars_names
        self.num_latent_vars = len(latent_vars_names)
        self.num_observed_vars = len(observed_vars_names)
        self.device = device

        self.born_machine = ClassicalBornMachine(num_latent_vars=self.num_latent_vars,
                                                 **born_machine_config).to(device)

        # Precompute all possible latent states (tuples)
        self.all_latent_states_tuples = generate_all_binary_outcomes(self.num_latent_vars)
        self.num_possible_latent_states = len(self.all_latent_states_tuples)

        # Partial function for the base kernel (Hamming kernel)
        self.base_kernel_func = partial(base_hamming_kernel_torch, 
                                        num_vars=self.num_latent_vars, 
                                        length_scale=base_kernel_length_scale)
        
        # Cache for score functions s_p(x,z) if x is fixed
        self._score_function_cache = {}


    def _get_precomputed_s_p(self, z_tuple, x_dict):
        # Simple caching for s_p(x,z) if x_dict is hashable (or use z_tuple as key if x is fixed for the run)
        # For now, assume x_dict is fixed throughout a KSD VI run, so z_tuple is the key
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
        """Precomputes s_p(x,z) for all z given a fixed x."""
        self._score_function_cache.clear()
        print("Precomputing score functions s_p(x,z)...")
        for z_tuple in self.all_latent_states_tuples:
            self._get_precomputed_s_p(z_tuple, x_dict) # Populates cache
        print("Score functions precomputed.")


    def train(self, x_observation_dict, num_epochs, lr_born_machine,
              verbose=True, true_posterior_for_tvd=None):
        
        if self.num_observed_vars > 0 and set(x_observation_dict.keys()) != set(self.observed_vars_names):
            raise ValueError("Keys in x_observation_dict must match self.observed_vars_names.")

        # Convert x_observation_dict to tensor for Born machine, keep dict for BN queries
        x_obs_list = [x_observation_dict[name] for name in self.observed_vars_names] if self.num_observed_vars > 0 else []
        x_obs_tensor_for_bm = torch.tensor(x_obs_list, dtype=torch.float32, device=self.device)
        
        born_machine_x_condition = None
        if self.born_machine.conditioning_dim > 0:
            if self.num_observed_vars == 0 :
                raise ValueError("Born machine is conditional but no observed vars specified for KSD.")
            if self.born_machine.conditioning_dim != self.num_observed_vars:
                 raise ValueError("Born machine conditioning_dim must match num_observed_vars if used for KSD.")
            born_machine_x_condition = x_obs_tensor_for_bm

        # Precompute all s_p(x,z) values for the fixed x_observation_dict
        self._precompute_all_s_p(x_observation_dict)

        optimizer_born = optim.Adam(self.born_machine.parameters(), lr=lr_born_machine)
        history = {'loss_ksd': [], 'tvd': []}

        # Precompute all (z_idx, z_jdx) pairs for faster lookup if needed
        # Not strictly necessary here as we iterate i, j up to num_possible_latent_states

        for epoch in range(num_epochs):
            optimizer_born.zero_grad()

            # Get current q_theta(z|x) for all z
            # Shape: (num_possible_latent_states,)
            q_probs_all_z = self.born_machine.get_probabilities(x_condition=born_machine_x_condition).squeeze()
            
            if q_probs_all_z.shape[0] != self.num_possible_latent_states:
                raise ValueError(f"Probabilities from Born machine have unexpected shape: {q_probs_all_z.shape}")

            # Calculate K_x(theta) = sum_{zi, zj} q(zi|x)q(zj|x) k_p(zi, zj | x)
            sum_k_p_weighted = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            for i in range(self.num_possible_latent_states):
                zi_tuple = self.all_latent_states_tuples[i]
                qi = q_probs_all_z[i]
                spi = self._score_function_cache[zi_tuple]

                for j in range(self.num_possible_latent_states):
                    zj_tuple = self.all_latent_states_tuples[j]
                    qj = q_probs_all_z[j]
                    spj = self._score_function_cache[zj_tuple]

                    kp_val = get_stein_kernel_kp_value(
                        zi_tuple, zj_tuple, x_observation_dict,
                        self.bn, self.latent_vars_names, self.observed_vars_names,
                        self.base_kernel_func,
                        spi, spj,
                        device=self.device
                    )
                    sum_k_p_weighted += qi * qj * kp_val
            
            K_x_theta = sum_k_p_weighted # This is the KSD^2 term

            # Objective is sqrt(K_x(theta))
            # Add a small epsilon for numerical stability if K_x_theta is very small or zero.
            loss = torch.sqrt(K_x_theta.clamp(min=1e-9)) 
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf KSD loss: {loss.item()}. K_x_theta: {K_x_theta.item()}. Skipping update.")
            else:
                loss.backward()
                optimizer_born.step()
            
            history['loss_ksd'].append(loss.item())

            if true_posterior_for_tvd is not None:
                current_q_dist_dict = self.born_machine.get_prob_dict(x_condition=born_machine_x_condition)
                tvd = calculate_tvd(true_posterior_for_tvd, current_q_dist_dict)
                history['tvd'].append(tvd)
            else:
                history['tvd'].append(np.nan)

            if verbose and (epoch % max(1, num_epochs // 20) == 0 or epoch == num_epochs - 1):
                log_msg = f"Epoch {epoch+1}/{num_epochs} | KSD Objective (sqrt): {loss.item():.4f}"
                if true_posterior_for_tvd and not np.isnan(history['tvd'][-1]):
                    log_msg += f" | TVD: {history['tvd'][-1]:.4f}"
                print(log_msg)
        
        return history
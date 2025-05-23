# ksd_vi_quantum.py
import torch
import torch.optim as optim
import numpy as np
from functools import partial

from bayesian_network import BayesianNetwork
from quantum_born_machine import QuantumBornMachine 
from stein_utils import (
    base_hamming_kernel_torch, 
    get_score_function_sp_for_z, 
    get_stein_kernel_kp_value
)
from utils import calculate_tvd, generate_all_binary_outcomes


class KSDVariationalInference:
    def __init__(self, 
                 bayesian_network: BayesianNetwork, 
                 latent_vars_names: list, 
                 observed_vars_names: list,
                 qbm_num_latent_vars: int, 
                 qbm_ansatz_layers: int = 1, 
                 qbm_conditioning_dim: int = 0, 
                 qbm_pennylane_device_name: str = "default.qubit", 
                 base_kernel_length_scale: float = 1.0, 
                 pytorch_device: str = 'cpu'):
        
        self.bn = bayesian_network
        self.latent_vars_names = latent_vars_names
        self.observed_vars_names = observed_vars_names
        self.num_latent_vars = qbm_num_latent_vars 
        self.num_observed_vars = len(observed_vars_names)
        self.pytorch_device = pytorch_device

        self.born_machine = QuantumBornMachine(
            num_latent_vars=self.num_latent_vars,
            ansatz_layers=qbm_ansatz_layers,
            conditioning_dim=qbm_conditioning_dim,
            device_name=qbm_pennylane_device_name 
        ).to(pytorch_device) 
        
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
            device=self.pytorch_device 
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
              verbose=True, true_posterior_for_tvd=None):
        
        if self.num_observed_vars > 0 and set(x_observation_dict.keys()) != set(self.observed_vars_names):
            raise ValueError("Keys in x_observation_dict must match self.observed_vars_names.")

        qbm_x_condition_input = None 
        if self.num_observed_vars > 0 and self.born_machine.conditioning_dim > 0:
            x_obs_list_for_qbm = [x_observation_dict[name] for name in self.observed_vars_names] 
            qbm_x_condition_input = torch.tensor(x_obs_list_for_qbm, dtype=torch.float32, device=self.pytorch_device)

        self._precompute_all_s_p(x_observation_dict) 

        optimizer_born = optim.Adam(self.born_machine.parameters(), lr=lr_born_machine)
        history = {'loss_ksd': [], 'tvd': []}

        for epoch in range(num_epochs):
            optimizer_born.zero_grad()

            q_probs_all_z = self.born_machine.get_probabilities(x_condition=qbm_x_condition_input).squeeze()
            q_probs_all_z = q_probs_all_z.to(torch.float64) # Ensure float64 for precision

            # --- DIAGNOSTIC PRINT FOR PROBABILITIES ---
            if verbose and epoch % (num_epochs // 10 if num_epochs >=10 else 1) == 0 :
                print(f"  Epoch {epoch+1} Q Probs (first 4): {q_probs_all_z.detach().cpu().numpy()[:4]}")
            # --- END DIAGNOSTIC ---

            if q_probs_all_z.shape[0] != self.num_possible_latent_states:
                raise ValueError(f"Probabilities from Born machine have unexpected shape: {q_probs_all_z.shape}, expected {self.num_possible_latent_states}")

            sum_k_p_weighted = torch.tensor(0.0, device=self.pytorch_device, dtype=torch.float64)

            for i in range(self.num_possible_latent_states):
                zi_tuple = self.all_latent_states_tuples[i]
                qi = q_probs_all_z[i] 
                spi = self._score_function_cache[zi_tuple].to(torch.float64)

                for j in range(self.num_possible_latent_states):
                    zj_tuple = self.all_latent_states_tuples[j]
                    qj = q_probs_all_z[j] 
                    spj = self._score_function_cache[zj_tuple].to(torch.float64)

                    kp_val = get_stein_kernel_kp_value(
                        zi_tuple, zj_tuple, x_observation_dict,
                        self.bn, self.latent_vars_names, self.observed_vars_names,
                        self.base_kernel_func,
                        spi, spj,
                        device=self.pytorch_device 
                    ).to(torch.float64)
                    sum_k_p_weighted += qi * qj * kp_val
            
            K_x_theta = sum_k_p_weighted 
            loss = torch.sqrt(K_x_theta.clamp(min=1e-12)) 
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf KSD loss: {loss.item()}. K_x_theta: {K_x_theta.item()}. Skipping update.")
            else:
                loss.backward()
                # --- DIAGNOSTIC PRINT FOR GRADIENTS ---
                if verbose and epoch % (num_epochs // 10 if num_epochs >=10 else 1) == 0:
                    grad_norm = None
                    if self.born_machine.theta.grad is not None:
                        grad_norm = self.born_machine.theta.grad.norm().item()
                    print(f"  Epoch {epoch+1} Theta Grad Norm: {grad_norm}")
                # --- END DIAGNOSTIC ---
                optimizer_born.step()
            
            history['loss_ksd'].append(loss.item())

            if true_posterior_for_tvd is not None:
                current_q_dist_dict = self.born_machine.get_prob_dict(x_condition=qbm_x_condition_input)
                tvd = calculate_tvd(true_posterior_for_tvd, current_q_dist_dict)
                history['tvd'].append(tvd)
            else:
                history['tvd'].append(np.nan)

            if verbose and (epoch % max(1, num_epochs // 20) == 0 or epoch == num_epochs - 1):
                log_msg = f"Epoch {epoch+1}/{num_epochs} | KSD Objective (sqrt): {loss.item():.6f}"
                if true_posterior_for_tvd and not np.isnan(history['tvd'][-1]):
                    log_msg += f" | TVD: {history['tvd'][-1]:.6f}"
                print(log_msg)
        
        return history
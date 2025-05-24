# adversarial_vi.py
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as nn_utils
import numpy as np
from bayesian_network import BayesianNetwork
from born_machine_classical_sim import ClassicalBornMachine
from classifier_pytorch import BinaryClassifierMLP
from utils import calculate_tvd, get_binary_key, get_outcome_tuple

class AdversarialVariationalInference:
    """
    Improved Adversarial Variational Inference with better optimization strategies.
    """
    def __init__(self, bayesian_network, latent_vars_names, observed_vars_names,
                 born_machine_config, classifier_config,
                 device='cpu'):
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

        classifier_input_dim = self.num_latent_vars
        if born_machine_config.get('conditioning_dim', 0) > 0:
            classifier_input_dim += born_machine_config['conditioning_dim']

        self.classifier = BinaryClassifierMLP(input_dim=classifier_input_dim, **classifier_config).to(device)

        self.prior_z_dist_dict = self.bn.get_prior_distribution(self.latent_vars_names)
        self.prior_z_outcomes = list(self.prior_z_dist_dict.keys())
        prior_probs_list = [self.prior_z_dist_dict[k] for k in self.prior_z_outcomes]
        
        if not prior_probs_list:
            self.prior_z_probs = torch.empty(0, dtype=torch.float32, device=self.device)
        else:
            self.prior_z_probs = torch.tensor(prior_probs_list,
                                              dtype=torch.float32, device=self.device)
            if self.prior_z_probs.numel() > 0 and not torch.isclose(torch.sum(self.prior_z_probs), torch.tensor(1.0)):
                self.prior_z_probs = self.prior_z_probs / torch.sum(self.prior_z_probs)

    def _sample_from_prior_z(self, num_samples):
        if self.num_latent_vars == 0:
            return torch.empty(num_samples, 0, device=self.device)
        if len(self.prior_z_outcomes) == 0 or self.prior_z_probs.numel() == 0:
            raise ValueError("Prior distribution p(z) is not properly initialized.")

        sampled_indices = torch.multinomial(self.prior_z_probs, num_samples, replacement=True)
        samples = torch.tensor([self.prior_z_outcomes[idx.item()] for idx in sampled_indices],
                               dtype=torch.float32, device=self.device)
        return samples

    def _get_log_p_x_given_z(self, x_obs_tensor, z_samples_tensor):
        log_p_x_obs_given_z_list = []
        x_obs_tuple = tuple(x_obs_tensor.cpu().long().tolist()) if self.num_observed_vars > 0 else ()

        for i in range(z_samples_tensor.shape[0]):
            z_sample_tuple = tuple(z_samples_tensor[i].cpu().long().tolist())
            
            current_assignment_dict_for_joint = {}
            if self.num_observed_vars > 0:
                for idx, name in enumerate(self.observed_vars_names):
                    current_assignment_dict_for_joint[name] = x_obs_tuple[idx]
            for idx, name in enumerate(self.latent_vars_names):
                current_assignment_dict_for_joint[name] = z_sample_tuple[idx]
            
            prob_x_obs_z_sample = 0.0
            other_bn_vars = [node for node in self.bn.nodes if node not in current_assignment_dict_for_joint]
            num_other_bn_vars = len(other_bn_vars)

            if num_other_bn_vars == 0:
                ordered_assign_tuple = [current_assignment_dict_for_joint[name] for name in self.bn.nodes]
                prob_x_obs_z_sample = self.bn.get_joint_probability(tuple(ordered_assign_tuple))
            else:
                from utils import generate_all_binary_outcomes
                for other_assignment in generate_all_binary_outcomes(num_other_bn_vars):
                    temp_full_assign_dict = dict(current_assignment_dict_for_joint)
                    for i_other, other_name in enumerate(other_bn_vars):
                        temp_full_assign_dict[other_name] = other_assignment[i_other]
                    
                    ordered_assign_tuple = [temp_full_assign_dict[name] for name in self.bn.nodes]
                    prob_x_obs_z_sample += self.bn.get_joint_probability(tuple(ordered_assign_tuple))

            prob_z_sample = self.prior_z_dist_dict.get(z_sample_tuple, 0.0)

            if prob_z_sample < 1e-9:
                if prob_x_obs_z_sample > 1e-9:
                    log_p_x_obs_given_z_list.append(torch.tensor(np.inf, dtype=torch.float32, device=self.device))
                else:
                    log_p_x_obs_given_z_list.append(torch.tensor(-np.inf, dtype=torch.float32, device=self.device))
            else:
                p_x_obs_given_z = prob_x_obs_z_sample / prob_z_sample
                log_p_x_obs_given_z_list.append(torch.log(torch.tensor(p_x_obs_given_z, dtype=torch.float32, device=self.device) + 1e-9))
        
        return torch.stack(log_p_x_obs_given_z_list)

    def train(self, x_observation_dict, num_epochs, batch_size, lr_born_machine, lr_classifier,
              k_classifier_steps=1, k_born_steps=1, verbose=True, true_posterior_for_tvd=None,
              use_lr_scheduler=True, gradient_clip_norm=10.0, baseline_decay=0.99,
              optimizer_type="adam", adam_betas=(0.9, 0.999)):
        
        if self.num_observed_vars > 0 and set(x_observation_dict.keys()) != set(self.observed_vars_names):
            raise ValueError("Keys in x_observation_dict must match self.observed_vars_names.")

        x_obs_list = [x_observation_dict[name] for name in self.observed_vars_names] if self.num_observed_vars > 0 else []
        x_obs_tensor = torch.tensor(x_obs_list, dtype=torch.float32, device=self.device)
        
        born_machine_x_condition = None
        if self.born_machine.conditioning_dim > 0:
            if self.num_observed_vars == 0:
                raise ValueError("Born machine is conditional but no observed vars specified.")
            if self.born_machine.conditioning_dim != self.num_observed_vars:
                raise ValueError("Born machine conditioning_dim must match num_observed_vars if used.")
            born_machine_x_condition = x_obs_tensor

        # Choose optimizers
        if optimizer_type == "adam":
            optimizer_born = optim.Adam(self.born_machine.parameters(), lr=lr_born_machine, betas=adam_betas)
            optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=lr_classifier, betas=adam_betas)
        else:
            optimizer_born = optim.SGD(self.born_machine.parameters(), lr=lr_born_machine, momentum=0.9)
            optimizer_classifier = optim.SGD(self.classifier.parameters(), lr=lr_classifier, momentum=0.9)

        # Learning rate schedulers
        scheduler_born = None
        scheduler_classifier = None
        if use_lr_scheduler:
            scheduler_born = optim.lr_scheduler.CosineAnnealingLR(optimizer_born, T_max=num_epochs, eta_min=lr_born_machine/10)
            scheduler_classifier = optim.lr_scheduler.CosineAnnealingLR(optimizer_classifier, T_max=num_epochs, eta_min=lr_classifier/10)

        criterion_classifier = nn.BCEWithLogitsLoss()

        # Running baseline for variance reduction in REINFORCE
        running_baseline = 0.0

        history = {'loss_classifier': [], 'loss_born_machine': [], 'tvd': [], 'grad_norm_born': [], 'grad_norm_classifier': []}
        
        best_tvd = float('inf')
        best_born_params = None
        best_classifier_params = None

        for epoch in range(num_epochs):
            # --- Train Classifier ---
            for _ in range(k_classifier_steps):
                optimizer_classifier.zero_grad()

                z_from_born = self.born_machine.sample(batch_size, x_condition=born_machine_x_condition)
                z_from_prior = self._sample_from_prior_z(batch_size)

                if self.classifier.network[0].in_features == self.num_latent_vars + self.num_observed_vars and self.num_observed_vars > 0:
                    x_obs_repeated = x_obs_tensor.unsqueeze(0).repeat(batch_size, 1)
                    inputs_born = torch.cat((z_from_born, x_obs_repeated), dim=1)
                    inputs_prior = torch.cat((z_from_prior, x_obs_repeated), dim=1)
                elif self.classifier.network[0].in_features == self.num_latent_vars:
                    inputs_born = z_from_born
                    inputs_prior = z_from_prior
                else:
                    raise ValueError("Classifier input dimension mismatch.")

                all_inputs = torch.cat((inputs_born, inputs_prior), dim=0)
                
                labels_born = torch.ones(batch_size, 1, device=self.device)
                labels_prior = torch.zeros(batch_size, 1, device=self.device)
                all_labels = torch.cat((labels_born, labels_prior), dim=0)

                logits_d = self.classifier(all_inputs)
                loss_d = criterion_classifier(logits_d, all_labels)
                
                loss_d.backward()
                
                # Gradient clipping for classifier
                grad_norm_d = nn_utils.clip_grad_norm_(self.classifier.parameters(), gradient_clip_norm)
                
                optimizer_classifier.step()
                
            history['loss_classifier'].append(loss_d.item())
            history['grad_norm_classifier'].append(grad_norm_d.item())

            # --- Train Born Machine ---
            for _ in range(k_born_steps):
                optimizer_born.zero_grad()

                z_q = self.born_machine.sample(batch_size, x_condition=born_machine_x_condition)

                if self.classifier.network[0].in_features == self.num_latent_vars + self.num_observed_vars and self.num_observed_vars > 0:
                    x_obs_rep_for_born_loss = x_obs_tensor.unsqueeze(0).repeat(batch_size, 1)
                    classifier_input_for_z_q = torch.cat((z_q, x_obs_rep_for_born_loss), dim=1)
                elif self.classifier.network[0].in_features == self.num_latent_vars:
                    classifier_input_for_z_q = z_q
                else:
                    raise ValueError("Classifier input dimension mismatch during Born machine training.")

                # Calculate terms for the REINFORCE objective with baseline
                logit_d_phi_z_q_values = self.classifier(classifier_input_for_z_q).squeeze()
                log_p_x_obs_given_z_q_values = self._get_log_p_x_given_z(x_obs_tensor, z_q)
                
                # Reward with baseline
                raw_reward = (logit_d_phi_z_q_values - log_p_x_obs_given_z_q_values)
                
                # Update running baseline
                if epoch == 0:
                    running_baseline = raw_reward.detach().mean().item()
                else:
                    running_baseline = baseline_decay * running_baseline + (1 - baseline_decay) * raw_reward.detach().mean().item()
                
                # Apply baseline
                reinforce_reward = raw_reward - running_baseline
                
                log_probs_q_of_z_q = self.born_machine.get_log_q_z_x(z_q, born_machine_x_condition)
                
                # Entropy regularization for exploration
                entropy_bonus = -0.01 * log_probs_q_of_z_q  # Small entropy bonus
                
                loss_q = (log_probs_q_of_z_q * reinforce_reward.detach() - entropy_bonus).mean()
                
                if torch.isnan(loss_q) or torch.isinf(loss_q):
                    print(f"Warning: NaN or Inf encountered in Born machine loss. Skipping update.")
                else:
                    loss_q.backward()
                    
                    # Gradient clipping for Born machine
                    grad_norm_q = nn_utils.clip_grad_norm_(self.born_machine.parameters(), gradient_clip_norm)
                    
                    optimizer_born.step()
                    
            history['loss_born_machine'].append(loss_q.item() if 'loss_q' in locals() and not (torch.isnan(loss_q) or torch.isinf(loss_q)) else np.nan)
            history['grad_norm_born'].append(grad_norm_q.item() if 'grad_norm_q' in locals() else 0.0)

            # Step schedulers
            if scheduler_born is not None:
                scheduler_born.step()
            if scheduler_classifier is not None:
                scheduler_classifier.step()

            # Calculate TVD
            if true_posterior_for_tvd is not None:
                current_q_dist_dict = self.born_machine.get_prob_dict(x_condition=born_machine_x_condition)
                tvd = calculate_tvd(true_posterior_for_tvd, current_q_dist_dict)
                history['tvd'].append(tvd)
                
                # Save best parameters
                if tvd < best_tvd:
                    best_tvd = tvd
                    best_born_params = self.born_machine.state_dict()
                    best_classifier_params = self.classifier.state_dict()
            else:
                history['tvd'].append(np.nan)

            if verbose and (epoch % max(1, num_epochs // 20) == 0 or epoch == num_epochs - 1):
                log_msg = f"Epoch {epoch+1}/{num_epochs} | Loss D: {loss_d.item():.4f} | Loss G: {history['loss_born_machine'][-1]:.4f}"
                if scheduler_born is not None:
                    log_msg += f" | LR_G: {scheduler_born.get_last_lr()[0]:.6f}"
                if true_posterior_for_tvd and not np.isnan(history['tvd'][-1]):
                    log_msg += f" | TVD: {history['tvd'][-1]:.4f}"
                print(log_msg)
        
        # Restore best parameters
        if best_born_params is not None and verbose:
            print(f"\nRestoring best parameters (TVD: {best_tvd:.6f})")
            self.born_machine.load_state_dict(best_born_params)
            self.classifier.load_state_dict(best_classifier_params)
        
        return history
# adversarial_vi.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from bayesian_network import BayesianNetwork
from born_machine_classical_sim import ClassicalBornMachine
from classifier_pytorch import BinaryClassifierMLP
from utils import calculate_tvd, get_binary_key, get_outcome_tuple

class AdversarialVariationalInference:
    """
    Implements Adversarial Variational Inference as described in the paper
    (Sec. IV A), using a classical Born machine simulation.
    Uses REINFORCE for the Born machine gradient.
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

        self.born_machine = ClassicalBornMachine(num_latent_vars=self.num_latent_vars,
                                                 **born_machine_config).to(device)

        classifier_input_dim = self.num_latent_vars
        if born_machine_config.get('conditioning_dim', 0) > 0 :
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
                print(f"Warning: Sum of prior_z_probs is {torch.sum(self.prior_z_probs)}. Re-normalizing.")
                self.prior_z_probs = self.prior_z_probs / torch.sum(self.prior_z_probs)


    def _sample_from_prior_z(self, num_samples):
        if self.num_latent_vars == 0:
             return torch.empty(num_samples, 0, device=self.device)
        if len(self.prior_z_outcomes) == 0 or self.prior_z_probs.numel() == 0:
            raise ValueError("Prior distribution p(z) is not properly initialized or empty for latent variables.")

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

            if prob_z_sample < 1e-9: # z_sample has approx zero prior prob
                 # if p(x,z) is also non-zero, then p(x|z) could be large.
                 # if p(x,z) is zero, then p(x|z) is 0/0 or 0/small.
                 # For safety, if p(z) is effectively zero, p(x|z) is problematic.
                 # However, the term is log p(x|z). If p(x|z) is zero, log p(x|z) = -inf.
                 # If p(x,z) is non-zero but p(z) is zero, then p(x|z) is infinite, log is +inf.
                 # This term is subtracted, so -log p(x|z) would be -inf.
                 # Let's assume if p(z) is zero, then p(x,z) must also be zero, so p(x|z) could be taken as 0.
                if prob_x_obs_z_sample > 1e-9: # p(x,z) > 0 but p(z) = 0. This shouldn't happen if BN is consistent.
                    # This implies p(x|z) is very large.
                    log_p_x_obs_given_z_list.append(torch.tensor(np.inf, dtype=torch.float32, device=self.device))
                else: # p(x,z) = 0 and p(z) = 0. Let p(x|z) = 0.
                    log_p_x_obs_given_z_list.append(torch.tensor(-np.inf, dtype=torch.float32, device=self.device))

            else:
                p_x_obs_given_z = prob_x_obs_z_sample / prob_z_sample
                log_p_x_obs_given_z_list.append(torch.log(torch.tensor(p_x_obs_given_z, dtype=torch.float32, device=self.device) + 1e-9))
        
        return torch.stack(log_p_x_obs_given_z_list)


    def train(self, x_observation_dict, num_epochs, batch_size, lr_born_machine, lr_classifier,
              k_classifier_steps=1, k_born_steps=1, verbose=True, true_posterior_for_tvd=None):
        if self.num_observed_vars > 0 and set(x_observation_dict.keys()) != set(self.observed_vars_names):
            raise ValueError("Keys in x_observation_dict must match self.observed_vars_names.")

        x_obs_list = [x_observation_dict[name] for name in self.observed_vars_names] if self.num_observed_vars > 0 else []
        x_obs_tensor = torch.tensor(x_obs_list, dtype=torch.float32, device=self.device)
        
        born_machine_x_condition = None
        if self.born_machine.conditioning_dim > 0:
            if self.num_observed_vars == 0 :
                raise ValueError("Born machine is conditional but no observed vars specified.")
            if self.born_machine.conditioning_dim != self.num_observed_vars:
                 raise ValueError("Born machine conditioning_dim must match num_observed_vars if used.")
            born_machine_x_condition = x_obs_tensor

        optimizer_born = optim.Adam(self.born_machine.parameters(), lr=lr_born_machine)
        optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=lr_classifier)
        criterion_classifier = nn.BCEWithLogitsLoss()

        history = {'loss_classifier': [], 'loss_born_machine': [], 'tvd': []}

        for epoch in range(num_epochs):
            # --- Train Classifier d_phi ---
            for _ in range(k_classifier_steps):
                optimizer_classifier.zero_grad()

                z_from_born = self.born_machine.sample(batch_size, x_condition=born_machine_x_condition)
                z_from_prior = self._sample_from_prior_z(batch_size)

                if self.classifier.network[0].in_features == self.num_latent_vars + self.num_observed_vars and self.num_observed_vars > 0 :
                    x_obs_repeated = x_obs_tensor.unsqueeze(0).repeat(batch_size, 1)
                    inputs_born = torch.cat((z_from_born, x_obs_repeated), dim=1)
                    inputs_prior = torch.cat((z_from_prior, x_obs_repeated), dim=1)
                elif self.classifier.network[0].in_features == self.num_latent_vars:
                     inputs_born = z_from_born
                     inputs_prior = z_from_prior
                else:
                    raise ValueError(
                        f"Classifier input dimension mismatch. Classifier expects {self.classifier.network[0].in_features}, "
                        f"latent_vars={self.num_latent_vars}, observed_vars={self.num_observed_vars}"
                    )

                all_inputs = torch.cat((inputs_born, inputs_prior), dim=0)
                
                labels_born = torch.ones(batch_size, 1, device=self.device)
                labels_prior = torch.zeros(batch_size, 1, device=self.device)
                all_labels = torch.cat((labels_born, labels_prior), dim=0)

                logits_d = self.classifier(all_inputs)
                loss_d = criterion_classifier(logits_d, all_labels)
                
                loss_d.backward()
                optimizer_classifier.step()
            history['loss_classifier'].append(loss_d.item())

            # --- Train Born Machine q_theta ---
            for _ in range(k_born_steps):
                optimizer_born.zero_grad()

                z_q = self.born_machine.sample(batch_size, x_condition=born_machine_x_condition)

                # Prepare classifier input for these samples z_q
                if self.classifier.network[0].in_features == self.num_latent_vars + self.num_observed_vars and self.num_observed_vars > 0:
                    x_obs_rep_for_born_loss = x_obs_tensor.unsqueeze(0).repeat(batch_size, 1)
                    classifier_input_for_z_q = torch.cat((z_q, x_obs_rep_for_born_loss), dim=1)
                elif self.classifier.network[0].in_features == self.num_latent_vars:
                    classifier_input_for_z_q = z_q
                else:
                    raise ValueError("Classifier input dimension mismatch during Born machine training.")

                # Calculate terms for the REINFORCE objective
                # Reward R(z_q) = logit[d_phi(z_q,x)] - log p(x|z_q)
                # logit[d_phi(z_q,x)] comes from the classifier (do not detach here, value needed for reward)
                logit_d_phi_z_q_values = self.classifier(classifier_input_for_z_q).squeeze()
                
                # log p(x|z_q)
                log_p_x_obs_given_z_q_values = self._get_log_p_x_given_z(x_obs_tensor, z_q)
                
                # The REINFORCE objective term (reward)
                # This term's value is based on current z_q and fixed classifier/BN for this step.
                reinforce_reward = (logit_d_phi_z_q_values - log_p_x_obs_given_z_q_values)
                
                # log q_theta(z_q|x_obs) - This is the term that carries gradient to Born machine params
                log_probs_q_of_z_q = self.born_machine.get_log_q_z_x(z_q, born_machine_x_condition)
                
                # Born machine loss: we want to minimize L_KL.
                # Gradient of L_KL is E[log_q * (Reward)]. So loss for minimization is (log_q * Reward.detach()).mean()
                # The .detach() on the reward ensures that gradients only flow through log_q_theta.
                loss_q = (log_probs_q_of_z_q * reinforce_reward.detach()).mean()
                
                # Handle potential NaN/Inf in loss_q before backward pass
                if torch.isnan(loss_q) or torch.isinf(loss_q):
                    print(f"Warning: NaN or Inf encountered in Born machine loss_q: {loss_q.item()}. Skipping backward pass for this step.")
                    print(f"  log_probs_q_of_z_q stats: mean={log_probs_q_of_z_q.mean().item()}, min={log_probs_q_of_z_q.min().item()}, max={log_probs_q_of_z_q.max().item()}")
                    print(f"  reinforce_reward stats: mean={reinforce_reward.mean().item()}, min={reinforce_reward.min().item()}, max={reinforce_reward.max().item()}")
                else:
                    loss_q.backward()
                    optimizer_born.step()
            history['loss_born_machine'].append(loss_q.item())


            if true_posterior_for_tvd is not None:
                current_q_dist_dict = self.born_machine.get_prob_dict(x_condition=born_machine_x_condition)
                tvd = calculate_tvd(true_posterior_for_tvd, current_q_dist_dict)
                history['tvd'].append(tvd)
            else:
                history['tvd'].append(np.nan)

            if verbose and (epoch % max(1, num_epochs // 20) == 0 or epoch == num_epochs - 1):
                log_msg = f"Epoch {epoch+1}/{num_epochs} | Loss D: {loss_d.item():.4f} | Loss G: {history['loss_born_machine'][-1]:.4f}"
                if true_posterior_for_tvd and not np.isnan(history['tvd'][-1]):
                    log_msg += f" | TVD: {history['tvd'][-1]:.4f}"
                print(log_msg)
        
        return history
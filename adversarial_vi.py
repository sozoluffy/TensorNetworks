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

    The goal is to make q_theta(z|x) approximate p_target(z|x).
    p_target(z|x) is the true posterior from a Bayesian Network.

    The optimization involves two parts:
    1. Training a classifier d_phi(z,x) to distinguish between:
       - Class 1: (z,x) where z ~ q_theta(z|x) and x ~ p_D(x) (data distribution)
       - Class 0: (z,x) where z ~ p_prior(z)   and x ~ p_D(x)
         (Here p_prior(z) is the prior over latent variables from the BN)
    2. Training the Born machine q_theta(z|x) to fool the classifier, by minimizing:
       L_KL(theta; phi) = E_{x~p_D(x)} E_{z~q_theta(z|x)} [logit(d_phi(z,x)) - log p_likelihood(x|z)]
       where logit(d) = log(d / (1-d)).
       And p_likelihood(x|z) is from the original Bayesian network: p(x|z) = p(x,z) / p(z).
       The paper uses p(z|x) = p(x|z)p(z)/p(x).
       The objective in Eq (3) is E[log(q(z|x)/p(z)) - log p(x|z)].
       The logit[d*] approximates log(q(z|x)/p(z)).
    """
    def __init__(self, bayesian_network, latent_vars_names, observed_vars_names,
                 born_machine_config, classifier_config,
                 device='cpu'):
        """
        Args:
            bayesian_network (BayesianNetwork): The BN defining the target posterior.
            latent_vars_names (list of str): Names of latent variables to infer.
            observed_vars_names (list of str): Names of observed variables to condition on.
                                              (Note: For simplicity in this initial version,
                                               we'll assume a single, fixed observation x_obs,
                                               so p_D(x) is a delta function at x_obs.
                                               Amortization over different x can be added later.)
            born_machine_config (dict): Config for ClassicalBornMachine.
                                        E.g., {'use_logits': True, 'conditioning_dim': 0}
                                        (conditioning_dim should be 0 if we fix x_obs for now)
            classifier_config (dict): Config for BinaryClassifierMLP.
                                      E.g., {'hidden_dims': [64,32]}
            device (str): 'cpu' or 'cuda'.
        """
        self.bn = bayesian_network
        self.latent_vars_names = latent_vars_names
        self.observed_vars_names = observed_vars_names # For p(x|z)
        self.num_latent_vars = len(latent_vars_names)
        self.num_observed_vars = len(observed_vars_names)
        self.device = device

        # --- Initialize Born Machine (q_theta(z|x_obs)) ---
        # For now, x_obs is fixed, so conditioning_dim for Born machine can be 0.
        # If we were to amortize over x, conditioning_dim would be num_observed_vars.
        # Let's assume born_machine_config['conditioning_dim'] reflects this.
        # If conditioning_dim > 0, the Born machine expects x_obs as input.
        self.born_machine = ClassicalBornMachine(num_latent_vars=self.num_latent_vars,
                                                 **born_machine_config).to(device)

        # --- Initialize Classifier (d_phi(z, x_obs)) ---
        # Classifier input: z (latent) and x_obs (observed).
        # If x_obs is fixed, its contribution to classifier input can be considered fixed.
        # The paper's d_phi(z,x) suggests x is an input.
        classifier_input_dim = self.num_latent_vars
        if born_machine_config.get('conditioning_dim', 0) > 0 : # If Born machine is conditional
             classifier_input_dim += born_machine_config['conditioning_dim']


        self.classifier = BinaryClassifierMLP(input_dim=classifier_input_dim, **classifier_config).to(device)

        # --- Precompute prior p(z) for latent variables ---
        # This is P(Z_L) from the BN, where Z_L are the latent variables.
        self.prior_z_dist_dict = self.bn.get_prior_distribution(self.latent_vars_names)
        self.prior_z_outcomes = list(self.prior_z_dist_dict.keys())
        self.prior_z_probs = torch.tensor([self.prior_z_dist_dict[k] for k in self.prior_z_outcomes],
                                          dtype=torch.float32, device=self.device)
        
        # For sampling from prior_z
        if not torch.isclose(torch.sum(self.prior_z_probs), torch.tensor(1.0)):
            print(f"Warning: Sum of prior_z_probs is {torch.sum(self.prior_z_probs)}. Re-normalizing.")
            self.prior_z_probs = self.prior_z_probs / torch.sum(self.prior_z_probs)


    def _sample_from_prior_z(self, num_samples):
        """Samples z ~ p_prior(z)."""
        if len(self.prior_z_outcomes) == 0 or self.prior_z_probs.numel() == 0:
            # This can happen if latent_vars_names is empty or prior calculation failed
            if self.num_latent_vars == 0: return torch.empty(num_samples, 0, device=self.device) # No latent vars
            raise ValueError("Prior distribution p(z) is not properly initialized or empty.")

        sampled_indices = torch.multinomial(self.prior_z_probs, num_samples, replacement=True)
        samples = torch.tensor([self.prior_z_outcomes[idx.item()] for idx in sampled_indices],
                               dtype=torch.float32, device=self.device)
        return samples # Shape (num_samples, num_latent_vars)

    def _get_log_p_x_given_z(self, x_obs_tensor, z_samples_tensor):
        """
        Calculates log p(x_obs | z) for each z in z_samples_tensor.
        p(x_obs | z) = p(x_obs, z) / p(z) from the Bayesian Network.

        Args:
            x_obs_tensor (torch.Tensor): The fixed observed variables, shape (num_observed_vars,).
            z_samples_tensor (torch.Tensor): Samples of latent variables, shape (batch_size, num_latent_vars).

        Returns:
            torch.Tensor: log p(x_obs | z) for each z, shape (batch_size,).
        """
        log_p_x_obs_given_z_list = []
        x_obs_tuple = tuple(x_obs_tensor.cpu().long().tolist())

        for i in range(z_samples_tensor.shape[0]):
            z_sample_tuple = tuple(z_samples_tensor[i].cpu().long().tolist())
            
            # Construct full assignment for p(x_obs, z)
            # This requires knowing which variable name corresponds to which index in x and z
            current_assignment_dict_for_joint = {}
            for idx, name in enumerate(self.observed_vars_names):
                current_assignment_dict_for_joint[name] = x_obs_tuple[idx]
            for idx, name in enumerate(self.latent_vars_names):
                current_assignment_dict_for_joint[name] = z_sample_tuple[idx]
            
            # Ensure all BN variables are covered if the BN is only over latent+observed
            # If BN has more variables, they need to be marginalized out for p(x_obs, z) and p(z).
            # For simplicity, assume here that latent_vars + observed_vars = all BN vars for this specific problem.
            # Or, that get_joint_probability and get_prior_distribution handle marginalization correctly.

            # Order for BN's get_joint_probability
            ordered_full_assignment_tuple = [0] * len(self.bn.nodes)
            all_involved_vars = set(self.latent_vars_names) | set(self.observed_vars_names)

            # If the BN has more variables than involved in the query (latent+observed),
            # then p(x|z) requires marginalizing out those other variables.
            # p(x|z) = sum_others P(x, z, others) / sum_others P(z, others)
            # This is complex. The paper's Eq (3) uses log p(x|z).
            # Let's assume p(x|z) can be directly computed or is given.
            # For a BN, p(x|z) is typically computed by P(x,z)/P(z).
            # P(z) is the prior P(Z_L) we precomputed.
            # P(x,z) is the joint P(Z_L, Z_O=x_obs).

            # Calculate P(x_obs, z_sample_tuple)
            # This requires summing over any variables in BN not in x_obs or z_sample_tuple
            vars_for_joint_xz = list(self.latent_vars_names) + list(self.observed_vars_names)
            
            # Create the assignment for P(x_obs, z_sample_tuple)
            assignment_for_joint_xz_dict = {}
            for idx, name in enumerate(self.latent_vars_names):
                assignment_for_joint_xz_dict[name] = z_sample_tuple[idx]
            for idx, name in enumerate(self.observed_vars_names):
                assignment_for_joint_xz_dict[name] = x_obs_tuple[idx]

            # We need to compute the probability of this partial assignment by marginalizing others
            prob_x_obs_z_sample = 0.0
            other_bn_vars = [node for node in self.bn.nodes if node not in assignment_for_joint_xz_dict]
            num_other_bn_vars = len(other_bn_vars)

            if num_other_bn_vars == 0: # All BN vars are covered by x_obs and z_sample
                ordered_assign_tuple = [assignment_for_joint_xz_dict[name] for name in self.bn.nodes]
                prob_x_obs_z_sample = self.bn.get_joint_probability(tuple(ordered_assign_tuple))
            else: # Marginalize
                from utils import generate_all_binary_outcomes # Already imported
                for other_assignment in generate_all_binary_outcomes(num_other_bn_vars):
                    temp_full_assign_dict = dict(assignment_for_joint_xz_dict)
                    for i_other, other_name in enumerate(other_bn_vars):
                        temp_full_assign_dict[other_name] = other_assignment[i_other]
                    
                    ordered_assign_tuple = [temp_full_assign_dict[name] for name in self.bn.nodes]
                    prob_x_obs_z_sample += self.bn.get_joint_probability(tuple(ordered_assign_tuple))

            # Get P(z_sample_tuple) from precomputed prior
            prob_z_sample = self.prior_z_dist_dict.get(z_sample_tuple, 0.0)

            if prob_z_sample < 1e-9: # Avoid division by zero if z_sample has 0 prior prob
                log_p_x_obs_given_z_list.append(torch.tensor(-np.inf, device=self.device)) # log(0) = -inf
            else:
                p_x_obs_given_z = prob_x_obs_z_sample / prob_z_sample
                log_p_x_obs_given_z_list.append(torch.log(torch.tensor(p_x_obs_given_z, device=self.device) + 1e-9))
        
        return torch.stack(log_p_x_obs_given_z_list)


    def train(self, x_observation_dict, num_epochs, batch_size, lr_born_machine, lr_classifier,
              k_classifier_steps=1, k_born_steps=1, verbose=True, true_posterior_for_tvd=None):
        """
        Performs the adversarial training.

        Args:
            x_observation_dict (dict): The fixed observation, e.g., {'W': 1}.
                                       Keys are observed_vars_names.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for sampling and training.
            lr_born_machine (float): Learning rate for the Born machine.
            lr_classifier (float): Learning rate for the classifier.
            k_classifier_steps (int): Number of optimization steps for classifier per epoch.
            k_born_steps (int): Number of optimization steps for Born machine per epoch.
            verbose (bool): If True, print progress.
            true_posterior_for_tvd (dict, optional): True posterior {outcome_tuple: prob} for TVD calculation.

        Returns:
            dict: Training history (losses, TVD).
        """
        if set(x_observation_dict.keys()) != set(self.observed_vars_names):
            raise ValueError("Keys in x_observation_dict must match self.observed_vars_names.")

        # Convert x_observation_dict to a tensor, ordered by self.observed_vars_names
        x_obs_list = [x_observation_dict[name] for name in self.observed_vars_names]
        x_obs_tensor = torch.tensor(x_obs_list, dtype=torch.float32, device=self.device)
         # x_obs_tensor will be used as conditioning if born machine is conditional.
        # It's also used as part of classifier input.

        # If Born machine is conditional, x_obs_tensor is its input.
        # If not, born_machine_x_condition is None.
        born_machine_x_condition = None
        if self.born_machine.conditioning_dim > 0:
            if self.born_machine.conditioning_dim != self.num_observed_vars:
                 raise ValueError("Born machine conditioning_dim must match num_observed_vars if used.")
            born_machine_x_condition = x_obs_tensor # Shape (num_observed_vars,)
            # The born machine's sample/get_probabilities will handle unsqueezing if needed.


        optimizer_born = optim.Adam(self.born_machine.parameters(), lr=lr_born_machine)
        optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=lr_classifier)
        criterion_classifier = nn.BCEWithLogitsLoss() # For classifier d_phi

        history = {'loss_classifier': [], 'loss_born_machine': [], 'tvd': []}

        for epoch in range(num_epochs):
            # --- Train Classifier d_phi ---
            for _ in range(k_classifier_steps):
                optimizer_classifier.zero_grad()

                # Samples z ~ q_theta(z|x_obs)
                z_from_born = self.born_machine.sample(batch_size, x_condition=born_machine_x_condition) # (batch, num_latent)
                
                # Samples z ~ p_prior(z)
                z_from_prior = self_sample_from_prior_z(batch_size) # (batch, num_latent)

                # Prepare classifier inputs and labels
                # Input to classifier: (z, x_obs_repeated)
                x_obs_repeated = x_obs_tensor.unsqueeze(0).repeat(batch_size, 1) # (batch, num_observed)

                # If born machine is conditional, its conditioning_dim > 0.
                # Classifier input dim was set up based on this.
                if self.born_machine.conditioning_dim > 0:
                    inputs_born = torch.cat((z_from_born, x_obs_repeated), dim=1)
                    inputs_prior = torch.cat((z_from_prior, x_obs_repeated), dim=1)
                else: # Born machine is not conditional on x_obs (e.g. x_obs is fixed and implicitly part of q_theta)
                      # Classifier might still take x_obs if its input_dim was set to include it.
                      # For now, let's assume classifier input is just z if born machine is not conditional.
                      # This needs to be consistent with classifier_input_dim initialization.
                      # Let's refine: classifier always gets z, and if x_obs is relevant (conditional model), it gets x_obs too.
                      # The current classifier_input_dim is num_latent_vars if cond_dim=0,
                      # and num_latent_vars + cond_dim if cond_dim > 0.
                      # This seems correct.
                    if self.classifier.network[0].in_features == self.num_latent_vars + self.num_observed_vars :
                         inputs_born = torch.cat((z_from_born, x_obs_repeated), dim=1)
                         inputs_prior = torch.cat((z_from_prior, x_obs_repeated), dim=1)
                    elif self.classifier.network[0].in_features == self.num_latent_vars:
                         inputs_born = z_from_born
                         inputs_prior = z_from_prior
                    else:
                        raise ValueError("Classifier input dimension mismatch.")


                all_inputs = torch.cat((inputs_born, inputs_prior), dim=0)
                
                labels_born = torch.ones(batch_size, 1, device=self.device)  # Class 1 for q_theta
                labels_prior = torch.zeros(batch_size, 1, device=self.device) # Class 0 for p_prior
                all_labels = torch.cat((labels_born, labels_prior), dim=0)

                logits_d = self.classifier(all_inputs)
                loss_d = criterion_classifier(logits_d, all_labels)
                
                loss_d.backward()
                optimizer_classifier.step()
            history['loss_classifier'].append(loss_d.item())

            # --- Train Born Machine q_theta ---
            for _ in range(k_born_steps):
                optimizer_born.zero_grad()

                # Samples z ~ q_theta(z|x_obs)
                z_q = self.born_machine.sample(batch_size, x_condition=born_machine_x_condition)

                # Prepare classifier input for these samples
                if self.born_machine.conditioning_dim > 0:
                    classifier_input_for_z_q = torch.cat((z_q, x_obs_tensor.unsqueeze(0).repeat(batch_size, 1)), dim=1)
                else:
                    if self.classifier.network[0].in_features == self.num_latent_vars + self.num_observed_vars :
                        classifier_input_for_z_q = torch.cat((z_q, x_obs_tensor.unsqueeze(0).repeat(batch_size, 1)), dim=1)
                    elif self.classifier.network[0].in_features == self.num_latent_vars:
                        classifier_input_for_z_q = z_q
                    else: # Should not happen if consistent
                        raise ValueError("Classifier input dimension mismatch during Born machine training.")


                # logit(d_phi(z,x)) term
                # d_phi(z,x) is P(class 1 | z,x) = sigmoid(logit_d(z,x))
                # logit_d(z,x) is the raw output of the classifier
                logit_d_phi_z_q = self.classifier(classifier_input_for_z_q).squeeze() # Shape (batch_size,)

                # log p_likelihood(x_obs|z) term
                log_p_x_obs_given_z_q = self._get_log_p_x_given_z(x_obs_tensor, z_q) # Shape (batch_size,)
                
                # Loss for Born machine (Eq. 7 in paper, minimizing this)
                # L_KL = E_{z~q} [ logit(d(z,x)) - log p(x|z) ]
                # The paper states gradient of log[q(z|x)/p(z)] w.r.t theta vanishes if d is optimal.
                # Here logit[d(z,x)] approximates log[q(z|x)/p(z)].
                # So the loss is E_{z~q} [ logit_d_phi_z_q - log_p_x_obs_given_z_q ]
                # Note: logit_d_phi_z_q itself depends on q_theta through z_q.
                # The paper's Appendix B simplifies gradient of E_q [log q - log p_target].
                # For REINFORCE-like gradients on E_q[f(z)], grad = E_q[f(z) * grad(log q(z))].
                # However, the paper uses a different objective based on the classifier's output.
                # If d_phi is optimal, logit(d_phi) = log(q/p_prior).
                # Loss = E_q [ log(q/p_prior) - log p(x|z) ]
                #      = E_q [ log q - log p_prior - log p(x|z) ]
                #      = E_q [ log q - (log p_prior + log p(x,z)/p(z)) ]
                #      = E_q [ log q - log p(x,z) ]
                # This is related to -ELBO if p(x,z) is unnormalized posterior.
                # Or, if p(x,z) is the true joint, then this is E_q [log q(z|x) - log p(x,z)].
                # The paper's objective (Eq. 7) is what we should implement:
                # L_KL(theta;phi) = E_{x~p_D(x)} E_{z~q_theta(z|x)} {logit[d_phi(z,x)] - log p(x|z)}
                # Since x is fixed to x_obs, E_{x~p_D(x)} drops.
                
                loss_q = (logit_d_phi_z_q - log_p_x_obs_given_z_q).mean()
                
                loss_q.backward() # This will propagate gradients through self.born_machine.params
                                  # via z_q and potentially through logit_d_phi_z_q if d_phi is not detached.
                                  # The paper assumes d_phi is fixed (optimal) when deriving grad_theta L_KL.
                                  # So, we should probably detach classifier output for Born machine loss.
                                  # L_KL(theta;phi_fixed) = E_{z~q_theta(z|x)} {logit[d_phi_fixed(z,x)] - log p(x|z)}
                                  # The gradient is then E [ (logit[d_phi_fixed] - log p(x|z)) * grad_theta log q_theta ]
                                  # Or, using parameter shift if applicable to ClassicalBornMachine.
                                  # For a generic ClassicalBornMachine with params being logits of a categorical dist,
                                  # PyTorch's autodiff on loss_q should work if log q_theta is implicitly handled.
                                  # self.born_machine.get_log_q_z_x(z_q, x_condition) would be log q_theta(z_q|x).
                                  # Let's use the direct objective from Eq (7) and rely on autodiff.
                                  # The term logit[d_phi(z,x)] is treated as a score for z_q.
                                  # If d_phi is part of the graph, its weights also get grads, which is not intended here.
                                  # So, we should use logit_d_phi_z_q.detach() if we follow Appendix B strictly.
                                  # However, some adversarial setups train generator with non-detached discriminator.
                                  # Let's try with attached first, as it's simpler. If unstable, use .detach().

                optimizer_born.step()
            history['loss_born_machine'].append(loss_q.item())

            # --- Logging and TVD ---
            if true_posterior_for_tvd is not None:
                current_q_dist_dict = self.born_machine.get_prob_dict(x_condition=born_machine_x_condition)
                # Ensure true_posterior_for_tvd keys are tuples of same latent var order
                # And current_q_dist_dict keys are also tuples of same latent var order
                # calculate_tvd needs consistent outcome representations.
                # self.born_machine.all_outcome_tuples gives the order for current_q_dist_dict.
                # true_posterior_for_tvd should also use this order for its keys.
                tvd = calculate_tvd(true_posterior_for_tvd, current_q_dist_dict)
                history['tvd'].append(tvd)
            else:
                history['tvd'].append(np.nan) # Or skip if not provided

            if verbose and (epoch % max(1, num_epochs // 20) == 0 or epoch == num_epochs - 1):
                log_msg = f"Epoch {epoch+1}/{num_epochs} | Loss D: {loss_d.item():.4f} | Loss G: {loss_q.item():.4f}"
                if true_posterior_for_tvd:
                    log_msg += f" | TVD: {history['tvd'][-1]:.4f}"
                print(log_msg)
        
        return history


if __name__ == '__main__':
    from bayesian_network import get_sprinkler_network
    from utils import plot_training_results

    print("--- Adversarial VI for Sprinkler Network P(C,S,R | W=1) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Setup Bayesian Network (Sprinkler)
    sprinkler_bn = get_sprinkler_network(random_cpts=False)
    latent_vars = ['C', 'S', 'R'] # We want to infer these
    observed_vars = ['W']         # Given W=1
    x_obs_dict_sprinkler = {'W': 1}

    # 2. Get True Posterior for TVD calculation
    # Keys of true_posterior_sprinkler will be tuples (c_val, s_val, r_val)
    true_posterior_sprinkler, p_obs = sprinkler_bn.get_true_posterior(latent_vars, x_obs_dict_sprinkler)
    print(f"True P(W=1) = {p_obs:.4f}")
    if p_obs == 0:
        print("Error: P(Observed) is zero. Cannot compute posterior.")
        exit()

    # 3. Configure Born Machine and Classifier
    # Sprinkler: 3 latent vars (C,S,R). 1 observed var (W).
    # For this test, Born machine is q(C,S,R | W=1), so W=1 is fixed.
    # We can make Born machine conditional on W if we want to amortize over W=0 and W=1.
    # Let's try unconditional q(C,S,R) first, where it implicitly learns for W=1.
    # Or, a conditional Born Machine q(C,S,R | W) and we always pass W=1.
    
    # Option A: Unconditional Born Machine (implicitly for W=1)
    # born_machine_cfg = {'use_logits': True, 'conditioning_dim': 0}
    # classifier_input_dim would be num_latent_vars (3 for C,S,R) + num_observed_vars (1 for W)
    # because d_phi(z,x) takes both.
    
    # Option B: Conditional Born Machine q(C,S,R | W)
    born_machine_cfg = {'use_logits': True, 'conditioning_dim': len(observed_vars)} # 1 for W
    # Classifier input dim is num_latent (3) + born_machine_cond_dim (1) = 4
    
    classifier_cfg = {'hidden_dims': [20, 10], 'use_batch_norm': False} # Small classifier

    # 4. Initialize AdversarialVI
    adv_vi = AdversarialVariationalInference(
        bayesian_network=sprinkler_bn,
        latent_vars_names=latent_vars,
        observed_vars_names=observed_vars,
        born_machine_config=born_machine_cfg,
        classifier_config=classifier_cfg,
        device=device
    )

    # 5. Train
    print("\nStarting training...")
    training_history = adv_vi.train(
        x_observation_dict=x_obs_dict_sprinkler,
        num_epochs=1000, # Paper uses 1000 for Fig 6
        batch_size=128,   # Paper uses 100 for Born machine, 100 for classifier samples
        lr_born_machine=0.005, # Paper uses 0.003 for Born machine
        lr_classifier=0.01,   # Paper uses 0.03 for MLP
        k_classifier_steps=2, # Train classifier a bit more
        k_born_steps=1,
        verbose=True,
        true_posterior_for_tvd=true_posterior_sprinkler
    )

    # 6. Plot results
    plot_training_results(training_history, "Adversarial VI - Sprinkler P(C,S,R | W=1)")

    # 7. Inspect final learned distribution
    final_q_dist = adv_vi.born_machine.get_prob_dict(
        x_condition=torch.tensor([x_obs_dict_sprinkler[name] for name in observed_vars], dtype=torch.float32, device=device)
        if born_machine_cfg['conditioning_dim'] > 0 else None
    )
    print("\n--- Comparison of True and Learned Posteriors ---")
    print(f"{'Outcome (C,S,R)':<15} | {'True P':<10} | {'Learned Q':<10} | Diff")
    print("-" * 60)
    sorted_outcomes = sorted(true_posterior_sprinkler.keys())
    for outcome_tuple in sorted_outcomes:
        true_p = true_posterior_sprinkler.get(outcome_tuple, 0.0)
        learned_q = final_q_dist.get(outcome_tuple, 0.0)
        diff = abs(true_p - learned_q)
        print(f"{str(outcome_tuple):<15} | {true_p:<10.4f} | {learned_q:<10.4f} | {diff:<10.4f}")

    final_tvd = calculate_tvd(true_posterior_sprinkler, final_q_dist)
    print(f"\nFinal TVD: {final_tvd:.4f}")


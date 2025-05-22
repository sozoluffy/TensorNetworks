# run_sprinkler_adversarial.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from bayesian_network import get_sprinkler_network
from adversarial_vi import AdversarialVariationalInference
from utils import plot_training_results, calculate_tvd

def run_sprinkler_experiment():
    """
    Runs the adversarial variational inference experiment for the Sprinkler network.
    Infers P(C,S,R | W=1).
    """
    print("--- Adversarial VI for Sprinkler Network P(C,S,R | W=1) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Setup Bayesian Network (Sprinkler)
    # Use fixed CPTs for reproducibility in this script
    sprinkler_bn = get_sprinkler_network(random_cpts=False) 
    
    latent_vars_names = ['C', 'S', 'R'] # We want to infer these: Cloudy, Sprinkler, Rain
    observed_vars_names = ['W']         # Given Grass Wet (W)
    
    # The specific observation we are conditioning on
    x_observation_dict = {'W': 1} # Grass is wet

    # 2. Get True Posterior for TVD calculation and comparison
    # The keys of true_posterior_sprinkler will be tuples (c_val, s_val, r_val)
    # in the order of latent_vars_names.
    true_posterior_dist, p_observed_true = sprinkler_bn.get_true_posterior(
        latent_vars_names, 
        x_observation_dict
    )
    
    print(f"True P(Observed={x_observation_dict}) = {p_observed_true:.4f}")
    if p_observed_true < 1e-9: # Check if evidence is impossible
        print("Error: P(Observed) is zero based on the Bayesian network. Cannot compute a valid posterior.")
        print("This might indicate an issue with CPTs or the chosen observation.")
        return

    print("True Posterior P(C,S,R | W=1):")
    for outcome, prob in sorted(true_posterior_dist.items()):
        print(f"  P({outcome} | W=1) = {prob:.4f}")


    # 3. Configure Born Machine and Classifier
    # The Born machine q_theta(z|x) will model P(C,S,R | W).
    # num_latent_vars = 3 (for C,S,R)
    # conditioning_dim for Born machine = 1 (for W)
    
    born_machine_config = {
        'use_logits': True, 
        'conditioning_dim': len(observed_vars_names) # Dimension of W
    }
    
    # The classifier d_phi(z,x) takes concatenated (z, x) as input.
    # Input dimension = num_latent_vars + conditioning_dim_of_born_machine
    # Example: z=(C,S,R), x=(W). Input to classifier is (C,S,R,W)
    classifier_config = {
        'hidden_dims': [32, 16], # Example hidden layer sizes for the MLP
        'use_batch_norm': False  # Optional: batch normalization
    }

    # 4. Initialize the AdversarialVariationalInference class
    adversarial_vi_model = AdversarialVariationalInference(
        bayesian_network=sprinkler_bn,
        latent_vars_names=latent_vars_names,
        observed_vars_names=observed_vars_names,
        born_machine_config=born_machine_config,
        classifier_config=classifier_config,
        device=device
    )

    # 5. Train the model
    print("\nStarting adversarial training...")
    num_epochs_train = 1500 # As per paper Fig 6, training can take a while
    batch_size_train = 100    # Paper used 100 for Born machine, 100 for classifier
    
    # Learning rates and training steps per epoch
    # These might need tuning. The paper mentions lr_born=0.003, lr_mlp=0.03.
    lr_born = 0.003
    lr_clf = 0.03
    k_clf_steps = 2 # Number of classifier updates per epoch
    k_born_steps = 1 # Number of Born machine updates per epoch

    training_history = adversarial_vi_model.train(
        x_observation_dict=x_observation_dict,
        num_epochs=num_epochs_train,
        batch_size=batch_size_train,
        lr_born_machine=lr_born,
        lr_classifier=lr_clf,
        k_classifier_steps=k_clf_steps,
        k_born_steps=k_born_steps,
        verbose=True,
        true_posterior_for_tvd=true_posterior_dist # Pass true posterior for TVD logging
    )

    # 6. Plot training results (losses, TVD over epochs)
    if training_history:
        plot_training_results(training_history, 
                              title=f"Adversarial VI - Sprinkler P({','.join(latent_vars_names)} | W=1)")

    # 7. Inspect and compare the final learned distribution with the true posterior
    print("\n--- Comparison of True Posterior and Final Learned Distribution ---")
    
    # Get the learned distribution q_theta(z | x_obs)
    # We need to provide the x_observation as a tensor for the conditional Born machine
    x_obs_tensor_for_bm = torch.tensor(
        [x_observation_dict[name] for name in observed_vars_names],
        dtype=torch.float32,
        device=device
    )
    
    learned_dist_dict = adversarial_vi_model.born_machine.get_prob_dict(
        x_condition=x_obs_tensor_for_bm if born_machine_config['conditioning_dim'] > 0 else None
    )

    print(f"{'Outcome ' + str(tuple(latent_vars_names)):<20} | {'True Posterior P(z|x)':<25} | {'Learned Q(z|x)':<20} | {'Difference |P-Q|':<20}")
    print("-" * 90)
    
    # Ensure outcomes are sorted for consistent display
    # The keys in true_posterior_dist and learned_dist_dict are tuples representing outcomes of latent_vars_names
    all_possible_latent_outcomes = sorted(list(true_posterior_dist.keys()))

    for outcome_tuple in all_possible_latent_outcomes:
        true_prob = true_posterior_dist.get(outcome_tuple, 0.0)
        learned_prob = learned_dist_dict.get(outcome_tuple, 0.0)
        diff = abs(true_prob - learned_prob)
        
        outcome_str = str(outcome_tuple)
        print(f"{outcome_str:<20} | {true_prob:<25.4f} | {learned_prob:<20.4f} | {diff:<20.4f}")

    final_tvd = calculate_tvd(true_posterior_dist, learned_dist_dict)
    print(f"\nFinal Total Variation Distance (TVD): {final_tvd:.4f}")

if __name__ == '__main__':
    run_sprinkler_experiment()

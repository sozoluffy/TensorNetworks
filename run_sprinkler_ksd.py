# run_sprinkler_ksd.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from bayesian_network import get_sprinkler_network
from ksd_vi import KSDVariationalInference # New import
from utils import plot_training_results, calculate_tvd

def run_sprinkler_ksd_experiment():
    print("--- KSD Variational Inference for Sprinkler Network P(C,S,R | W=1) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    sprinkler_bn = get_sprinkler_network(random_cpts=False) 
    
    latent_vars_names = ['C', 'S', 'R']
    observed_vars_names = ['W']
    x_observation_dict = {'W': 1}

    true_posterior_dist, p_observed_true = sprinkler_bn.get_true_posterior(
        latent_vars_names, 
        x_observation_dict
    )
    
    print(f"True P(Observed={x_observation_dict}) = {p_observed_true:.4f}")
    if p_observed_true < 1e-9:
        print("Error: P(Observed) is zero. Cannot compute a valid posterior.")
        return

    print("True Posterior P(C,S,R | W=1):")
    for outcome, prob in sorted(true_posterior_dist.items()):
        print(f"  P({outcome} | W=1) = {prob:.4f}")

    # Born Machine configuration
    born_machine_config = {
        'use_logits': True, 
        'conditioning_dim': len(observed_vars_names) # Conditional on W
    }
    
    # KSD VI Model Initialization
    ksd_vi_model = KSDVariationalInference(
        bayesian_network=sprinkler_bn,
        latent_vars_names=latent_vars_names,
        observed_vars_names=observed_vars_names,
        born_machine_config=born_machine_config,
        base_kernel_length_scale=1.0, # For Hamming kernel exp(-dist/n)
        device=device
    )

    print("\nStarting KSD training...")
    # The paper notes KSD can be slower or achieve suboptimal results for small examples
    # compared to their KL objective (Sec V.A).
    # "Optimizing the KL objective is faster than the KSD objective."
    # "Born machines trained with the KSD objective yield suboptimal results in comparison to the KL objective."
    # Let's try similar hyperparameters to the adversarial one first.
    num_epochs_train = 1000 # Paper Fig 6 uses 1000 epochs
    lr_born = 0.003       # Paper mentions 0.003 for Born machine with KL

    training_history = ksd_vi_model.train(
        x_observation_dict=x_observation_dict,
        num_epochs=num_epochs_train,
        lr_born_machine=lr_born,
        verbose=True,
        true_posterior_for_tvd=true_posterior_dist
    )

    if training_history:
        plot_training_results(training_history, 
                              title=f"KSD VI - Sprinkler P({','.join(latent_vars_names)} | W=1)")

    print("\n--- Comparison of True Posterior and Final Learned Distribution (KSD) ---")
    x_obs_tensor_for_bm = torch.tensor(
        [x_observation_dict[name] for name in observed_vars_names],
        dtype=torch.float32,
        device=device
    ) if ksd_vi_model.num_observed_vars > 0 else None
    
    learned_dist_dict = ksd_vi_model.born_machine.get_prob_dict(
        x_condition=x_obs_tensor_for_bm
    )

    print(f"{'Outcome ' + str(tuple(latent_vars_names)):<20} | {'True Posterior P(z|x)':<25} | {'Learned Q(z|x)':<20} | {'Difference |P-Q|':<20}")
    print("-" * 90)
    
    all_possible_latent_outcomes = sorted(list(true_posterior_dist.keys()))

    for outcome_tuple in all_possible_latent_outcomes:
        true_prob = true_posterior_dist.get(outcome_tuple, 0.0)
        learned_prob = learned_dist_dict.get(outcome_tuple, 0.0)
        diff = abs(true_prob - learned_prob)
        outcome_str = str(outcome_tuple)
        print(f"{outcome_str:<20} | {true_prob:<25.4f} | {learned_prob:<20.4f} | {diff:<20.4f}")

    final_tvd = calculate_tvd(true_posterior_dist, learned_dist_dict)
    print(f"\nFinal Total Variation Distance (TVD) with KSD: {final_tvd:.4f}")

if __name__ == '__main__':
    run_sprinkler_ksd_experiment()
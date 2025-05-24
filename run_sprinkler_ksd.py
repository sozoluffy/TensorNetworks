# run_sprinkler_ksd.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from bayesian_network import get_sprinkler_network
from ksd_vi import KSDVariationalInference
from utils import plot_training_results, calculate_tvd

def run_sprinkler_ksd_experiment():
    print("--- Improved KSD Variational Inference for Sprinkler Network P(C,S,R | W=1) ---")
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
        print("Error: P(Observed) is zero.")
        return

    # Enhanced Born Machine configuration
    born_machine_config = {
        'use_logits': True,
        'conditioning_dim': len(observed_vars_names),
        'init_method': 'uniform',  # Try uniform initialization for KSD
        'hidden_dims': None,       # Use default for simple problem
        'use_layer_norm': False    # Simpler architecture for small problem
    }
    
    # KSD VI Model with improvements
    ksd_vi_model = KSDVariationalInference(
        bayesian_network=sprinkler_bn,
        latent_vars_names=latent_vars_names,
        observed_vars_names=observed_vars_names,
        born_machine_config=born_machine_config,
        base_kernel_length_scale=1.0,
        device=device
    )

    print("\nStarting improved KSD training...")
    
    # Refined hyperparameters for KSD
    num_epochs_train = 2000      # More epochs to find optimum
    lr_born = 0.003              # Lower initial LR for KSD
    use_lr_scheduler = True      # Keep cosine annealing
    gradient_clip_norm = 5.0     # Tighter gradient clipping
    optimizer_type = "adam"      # Adam optimizer
    adam_betas = (0.9, 0.999)
    entropy_weight = 0.001       # Much smaller entropy weight
    patience = 200               # Early stopping patience

    training_history = ksd_vi_model.train(
        x_observation_dict=x_observation_dict,
        num_epochs=num_epochs_train,
        lr_born_machine=lr_born,
        verbose=True,
        true_posterior_for_tvd=true_posterior_dist,
        use_lr_scheduler=use_lr_scheduler,
        gradient_clip_norm=gradient_clip_norm,
        optimizer_type=optimizer_type,
        adam_betas=adam_betas,
        entropy_weight=entropy_weight,
        patience=patience
    )

    # Enhanced plotting
    if training_history:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        epochs = range(len(training_history['loss_ksd']))
        
        # Mark best epoch
        best_epoch_idx = None
        best_tvd_value = None
        if training_history['tvd']:
            best_epoch_idx = np.argmin(training_history['tvd'])
            best_tvd_value = training_history['tvd'][best_epoch_idx]
        
        # KSD loss
        axes[0].plot(epochs, training_history['loss_ksd'], 'b-', label='KSD Loss')
        axes[0].set_ylabel('KSD Loss')
        axes[0].set_yscale('log')
        if best_epoch_idx is not None:
            axes[0].axvline(x=best_epoch_idx, color='r', linestyle='--', alpha=0.5, label=f'Best TVD @ epoch {best_epoch_idx+1}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # TVD
        axes[1].plot(epochs, training_history['tvd'], 'r-', label='TVD')
        axes[1].set_ylabel('Total Variation Distance')
        if best_epoch_idx is not None:
            axes[1].axvline(x=best_epoch_idx, color='r', linestyle='--', alpha=0.5)
            axes[1].axhline(y=best_tvd_value, color='g', linestyle=':', alpha=0.5, label=f'Best TVD = {best_tvd_value:.6f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Gradient norm
        axes[2].plot(epochs, training_history['grad_norm'], 'g-', label='Gradient Norm')
        axes[2].set_ylabel('Gradient Norm')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Entropy
        axes[3].plot(epochs, training_history['entropy'], 'm-', label='Entropy')
        axes[3].set_ylabel('Entropy')
        axes[3].set_xlabel('Epoch')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        fig.suptitle(f"Improved KSD VI - Sprinkler P({','.join(latent_vars_names)} | W=1)", fontsize=16)
        plt.tight_layout()
        plt.show()

    print("\n--- Final Results (Using Best Parameters) ---")
    
    x_obs_tensor_for_bm = torch.tensor(
        [x_observation_dict[name] for name in observed_vars_names],
        dtype=torch.float32,
        device=device
    ) if ksd_vi_model.num_observed_vars > 0 else None
    
    # Get the final learned distribution
    learned_dist_dict = ksd_vi_model.born_machine.get_prob_dict(
        x_condition=x_obs_tensor_for_bm
    )

    print(f"{'Outcome':<20} | {'True P(z|x)':<15} | {'Learned Q(z|x)':<15} | {'Difference':<15}")
    print("-" * 70)
    
    all_possible_latent_outcomes = sorted(list(true_posterior_dist.keys()))
    max_diff = 0.0

    for outcome_tuple in all_possible_latent_outcomes:
        true_prob = true_posterior_dist.get(outcome_tuple, 0.0)
        learned_prob = learned_dist_dict.get(outcome_tuple, 0.0)
        diff = abs(true_prob - learned_prob)
        max_diff = max(max_diff, diff)
        outcome_str = str(outcome_tuple)
        print(f"{outcome_str:<20} | {true_prob:<15.6f} | {learned_prob:<15.6f} | {diff:<15.6f}")

    # Recalculate TVD to verify
    final_tvd_verified = calculate_tvd(true_posterior_dist, learned_dist_dict)
    best_tvd = min(training_history['tvd']) if training_history['tvd'] else final_tvd_verified
    
    print(f"\nFinal TVD (verified): {final_tvd_verified:.6f}")
    print(f"Best TVD from history: {best_tvd:.6f}")
    print(f"Max pointwise difference: {max_diff:.6f}")
    
    # Find when best was achieved
    if training_history['tvd']:
        best_epoch_idx = np.argmin(training_history['tvd'])
        print(f"Best performance at epoch: {best_epoch_idx + 1}")
        
        # Sanity check
        if abs(final_tvd_verified - best_tvd) > 0.01:
            print(f"\nWARNING: Final TVD ({final_tvd_verified:.6f}) differs significantly from expected best ({best_tvd:.6f})")
            print("This indicates a parameter restoration issue.")
    
    # Additional analysis
    print("\n--- Training Statistics ---")
    if training_history['tvd']:
        tvd_array = np.array(training_history['tvd'])
        print(f"Mean TVD: {np.mean(tvd_array):.6f}")
        print(f"Std TVD: {np.std(tvd_array):.6f}")
        print(f"Min TVD: {np.min(tvd_array):.6f}")
        print(f"Final 100 epochs mean TVD: {np.mean(tvd_array[-100:]):.6f}")
        
        # Check stability in final epochs
        if len(tvd_array) > 200:
            early_std = np.std(tvd_array[:100])
            late_std = np.std(tvd_array[-100:])
            print(f"\nStability Analysis:")
            print(f"Early training std (first 100 epochs): {early_std:.6f}")
            print(f"Late training std (last 100 epochs): {late_std:.6f}")
            
            if late_std > early_std * 2:
                print("Warning: Training became less stable over time.")
    
    # Plot probability comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # True posterior
    true_probs = [true_posterior_dist[outcome] for outcome in all_possible_latent_outcomes]
    outcome_labels = [str(outcome) for outcome in all_possible_latent_outcomes]
    x_pos = np.arange(len(outcome_labels))
    
    ax1.bar(x_pos, true_probs, alpha=0.7, label='True Posterior')
    ax1.set_xlabel('Outcome (C,S,R)')
    ax1.set_ylabel('Probability')
    ax1.set_title('True Posterior P(C,S,R | W=1)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(outcome_labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Learned posterior
    learned_probs = [learned_dist_dict[outcome] for outcome in all_possible_latent_outcomes]
    
    width = 0.35
    ax2.bar(x_pos - width/2, true_probs, width, alpha=0.7, label='True')
    ax2.bar(x_pos + width/2, learned_probs, width, alpha=0.7, label='Learned')
    ax2.set_xlabel('Outcome (C,S,R)')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'Posterior Comparison (TVD = {final_tvd_verified:.6f})')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(outcome_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_sprinkler_ksd_experiment()
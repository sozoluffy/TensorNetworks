# run_sprinkler_adversarial.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from bayesian_network import get_sprinkler_network
from adversarial_vi import AdversarialVariationalInference
from utils import plot_training_results, calculate_tvd

def run_sprinkler_experiment():
    """
    Runs the improved adversarial variational inference experiment with better stability.
    """
    print("--- Improved Adversarial VI for Sprinkler Network P(C,S,R | W=1) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Setup Bayesian Network
    sprinkler_bn = get_sprinkler_network(random_cpts=False)
    
    latent_vars_names = ['C', 'S', 'R']
    observed_vars_names = ['W']
    x_observation_dict = {'W': 1}

    # Get true posterior
    true_posterior_dist, p_observed_true = sprinkler_bn.get_true_posterior(
        latent_vars_names,
        x_observation_dict
    )
    
    print(f"True P(Observed={'W':1})={p_observed_true:.4f}")
    if p_observed_true < 1e-9:
        print("Error: P(Observed) is zero.")
        return

    # Refined configuration for stability
    born_machine_config = {
        'use_logits': True,
        'conditioning_dim': len(observed_vars_names),
        'init_method': 'uniform'  # Start from uniform distribution
    }
    
    classifier_config = {
        'hidden_dims': [32, 16],  # Smaller classifier for stability
        'use_batch_norm': False   # Simpler without batch norm for small problem
    }

    # Initialize model
    adversarial_vi_model = AdversarialVariationalInference(
        bayesian_network=sprinkler_bn,
        latent_vars_names=latent_vars_names,
        observed_vars_names=observed_vars_names,
        born_machine_config=born_machine_config,
        classifier_config=classifier_config,
        device=device
    )

    # Refined training parameters for stability
    print("\nStarting improved adversarial training...")
    num_epochs_train = 1500      # Fewer epochs to avoid overtraining
    batch_size_train = 100       # Smaller batch size
    lr_born = 0.003              # Lower learning rate
    lr_clf = 0.03                # Lower classifier LR
    k_clf_steps = 5              # More classifier updates for stability
    k_born_steps = 1
    
    # Stability parameters
    use_lr_scheduler = True
    gradient_clip_norm = 5.0     # Tighter clipping
    baseline_decay = 0.95        # Faster baseline adaptation
    optimizer_type = "adam"
    adam_betas = (0.5, 0.999)    # Lower beta1 for adversarial training

    training_history = adversarial_vi_model.train(
        x_observation_dict=x_observation_dict,
        num_epochs=num_epochs_train,
        batch_size=batch_size_train,
        lr_born_machine=lr_born,
        lr_classifier=lr_clf,
        k_classifier_steps=k_clf_steps,
        k_born_steps=k_born_steps,
        verbose=True,
        true_posterior_for_tvd=true_posterior_dist,
        use_lr_scheduler=use_lr_scheduler,
        gradient_clip_norm=gradient_clip_norm,
        baseline_decay=baseline_decay,
        optimizer_type=optimizer_type,
        adam_betas=adam_betas
    )

    # Enhanced plotting with stability analysis
    if training_history:
        fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        epochs = range(len(training_history['loss_classifier']))
        
        # Find best epoch
        best_epoch_idx = np.argmin(training_history['tvd'])
        best_tvd_value = training_history['tvd'][best_epoch_idx]
        
        # Classifier loss
        axes[0].plot(epochs, training_history['loss_classifier'], 'b-', label='Classifier Loss', alpha=0.7)
        axes[0].axvline(x=best_epoch_idx, color='r', linestyle='--', alpha=0.5, label=f'Best TVD @ epoch {best_epoch_idx+1}')
        axes[0].set_ylabel('Classifier Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Born machine loss
        axes[1].plot(epochs, training_history['loss_born_machine'], 'g-', label='Born Machine Loss', alpha=0.7)
        axes[1].axvline(x=best_epoch_idx, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Born Machine Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # TVD with best point marked
        axes[2].plot(epochs, training_history['tvd'], 'r-', label='TVD')
        axes[2].axvline(x=best_epoch_idx, color='r', linestyle='--', alpha=0.5)
        axes[2].axhline(y=best_tvd_value, color='g', linestyle=':', alpha=0.5, label=f'Best TVD = {best_tvd_value:.6f}')
        axes[2].set_ylabel('Total Variation Distance')
        axes[2].set_ylim(0, max(0.1, max(training_history['tvd'])))  # Zoom in on relevant range
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Gradient norms
        axes[3].plot(epochs, training_history['grad_norm_born'], 'g-', label='Born Grad Norm', alpha=0.7)
        axes[3].plot(epochs, training_history['grad_norm_classifier'], 'b-', label='Classifier Grad Norm', alpha=0.7)
        axes[3].set_ylabel('Gradient Norm')
        axes[3].set_yscale('log')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Stability metric: rolling std of TVD
        window_size = 50
        if len(training_history['tvd']) > window_size:
            tvd_array = np.array(training_history['tvd'])
            rolling_std = np.array([np.std(tvd_array[max(0, i-window_size):i+1])
                                   for i in range(len(tvd_array))])
            axes[4].plot(epochs, rolling_std, 'm-', label=f'TVD Stability (std over {window_size} epochs)')
            axes[4].axvline(x=best_epoch_idx, color='r', linestyle='--', alpha=0.5)
            axes[4].set_ylabel('TVD Rolling Std')
            axes[4].set_xlabel('Epoch')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
        
        fig.suptitle(f"Improved Adversarial VI - Sprinkler P({','.join(latent_vars_names)} | W=1)", fontsize=16)
        plt.tight_layout()
        plt.show()

    # Final comparison (using best parameters)
    print("\n--- Final Results (Using Best Parameters) ---")
    
    x_obs_tensor_for_bm = torch.tensor(
        [x_observation_dict[name] for name in observed_vars_names],
        dtype=torch.float32,
        device=device
    )
    
    learned_dist_dict = adversarial_vi_model.born_machine.get_prob_dict(
        x_condition=x_obs_tensor_for_bm if born_machine_config['conditioning_dim'] > 0 else None
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

    final_tvd = calculate_tvd(true_posterior_dist, learned_dist_dict)
    print(f"\nFinal TVD (with best parameters): {final_tvd:.6f}")
    print(f"Best TVD achieved during training: {min(training_history['tvd']):.6f}")
    print(f"Max pointwise difference: {max_diff:.6f}")
    print(f"Best performance at epoch: {best_epoch_idx + 1}")
    
    # Analyze stability
    if len(training_history['tvd']) > 100:
        early_tvd_std = np.std(training_history['tvd'][:100])
        late_tvd_std = np.std(training_history['tvd'][-100:])
        print(f"\nStability Analysis:")
        print(f"Early training TVD std (first 100 epochs): {early_tvd_std:.6f}")
        print(f"Late training TVD std (last 100 epochs): {late_tvd_std:.6f}")

    # Additional analysis
    print("\n--- Training Statistics ---")
    # Recalculate TVD to verify
    final_tvd_verified = calculate_tvd(true_posterior_dist, learned_dist_dict)
    best_tvd = min(training_history['tvd']) if training_history['tvd'] else final_tvd_verified
    
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
    run_sprinkler_experiment()

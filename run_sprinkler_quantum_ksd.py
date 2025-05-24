# run_sprinkler_quantum_ksd.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from bayesian_network import get_sprinkler_network
from ksd_vi_quantum import KSDVariationalInference
from utils import plot_training_results, calculate_tvd


def run_sprinkler_quantum_ksd_experiment():
    print("--- KSD Variational Inference with Improved QuantumBornMachine for Sprinkler P(C,S,R | W=1) ---")
    pytorch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using PyTorch device: {pytorch_device}")

    pennylane_device_name_for_qbm = "default.qubit"

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
        print("Error: P(Observed) is zero based on the BN.")
        return

    qbm_num_latent_vars = len(latent_vars_names)
    
    # --- IMPROVED HYPERPARAMETERS ---
    qbm_ansatz_layers = 4      # Increased from 2
    qbm_ansatz_type = "hardware_efficient"  # Better ansatz
    qbm_init_method = "small_random"  # Better initialization
    lr_born = 0.005            # Slightly higher initial LR (will decay)
    num_epochs_train = 1000    # More epochs
    use_lr_scheduler = True    # Enable cosine annealing
    gradient_clip_norm = 10.0  # Gradient clipping
    optimizer_type = "adam"    # Adam optimizer
    adam_betas = (0.9, 0.999)  # Adam hyperparameters
    
    experiment_label = f"{qbm_ansatz_type}_L{qbm_ansatz_layers}_LR{lr_born}_scheduler"
    # --- END HYPERPARAMETERS ---
    
    qbm_conditioning_dim = 0

    ksd_vi_model = KSDVariationalInference(
        bayesian_network=sprinkler_bn,
        latent_vars_names=latent_vars_names,
        observed_vars_names=observed_vars_names,
        qbm_num_latent_vars=qbm_num_latent_vars,
        qbm_ansatz_layers=qbm_ansatz_layers,
        qbm_conditioning_dim=qbm_conditioning_dim,
        qbm_pennylane_device_name=pennylane_device_name_for_qbm,
        qbm_ansatz_type=qbm_ansatz_type,
        qbm_init_method=qbm_init_method,
        base_kernel_length_scale=1.0,
        pytorch_device=pytorch_device
    )
    
    print(f"\nINFO: Experiment: {experiment_label}")
    print(f"INFO: QuantumBornMachine initialized with:")
    print(f"  - {qbm_num_latent_vars} qubits")
    print(f"  - {qbm_ansatz_layers} ansatz layers")
    print(f"  - Ansatz type: {qbm_ansatz_type}")
    print(f"  - Initialization: {qbm_init_method}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in ksd_vi_model.born_machine.parameters() if p.requires_grad)}")
    print(f"  - Learning rate: {lr_born} (with cosine annealing)")
    print(f"  - Gradient clipping: {gradient_clip_norm}")
    print(f"  - Epochs: {num_epochs_train}")

    print("\nStarting KSD training with Quantum Born Machine...")
    
    training_history = ksd_vi_model.train(
        x_observation_dict=x_observation_dict,
        num_epochs=num_epochs_train,
        lr_born_machine=lr_born,
        verbose=True,
        true_posterior_for_tvd=true_posterior_dist,
        use_lr_scheduler=use_lr_scheduler,
        gradient_clip_norm=gradient_clip_norm,
        optimizer_type=optimizer_type,
        adam_betas=adam_betas
    )

    if training_history:
        plot_title = (f"KSD VI (QBM {experiment_label}) - Sprinkler "
                      f"P({','.join(latent_vars_names)} | W=1)")
        
        # Enhanced plotting with gradient norms
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        epochs = range(len(training_history['loss_ksd']))
        
        # Loss plot
        axes[0].plot(epochs, training_history['loss_ksd'], 'b-', label='KSD Loss')
        axes[0].set_ylabel('KSD Loss')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # TVD plot
        axes[1].plot(epochs, training_history['tvd'], 'r-', label='TVD')
        axes[1].set_ylabel('Total Variation Distance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Gradient norm plot
        axes[2].plot(epochs, training_history['grad_norm'], 'g-', label='Gradient Norm')
        axes[2].set_ylabel('Gradient Norm')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(plot_title, fontsize=16)
        plt.tight_layout()
        plt.show()

    print(f"\n--- Final Results (KSD QBM - {experiment_label}) ---")
    
    qbm_x_cond_input_for_probs = None
    if ksd_vi_model.num_observed_vars > 0 and ksd_vi_model.born_machine.conditioning_dim > 0:
        x_obs_list_for_qbm_probs = [x_observation_dict[name] for name in observed_vars_names]
        qbm_x_cond_input_for_probs = torch.tensor(x_obs_list_for_qbm_probs, dtype=torch.float32, device=pytorch_device)

    learned_dist_dict = ksd_vi_model.born_machine.get_prob_dict(x_condition=qbm_x_cond_input_for_probs)

    print(f"{'Outcome ' + str(tuple(latent_vars_names)):<20} | {'True P(z|x)':<15} | {'Learned Q(z|x)':<15} | {'Difference':<15}")
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
    print(f"\nFinal TVD: {final_tvd:.6f}")
    print(f"Max pointwise difference: {max_diff:.6f}")
    print(f"Best TVD achieved during training: {min(training_history['tvd']):.6f}")


if __name__ == '__main__':
    run_sprinkler_quantum_ksd_experiment()
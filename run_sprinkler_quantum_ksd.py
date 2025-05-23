# run_sprinkler_quantum_ksd.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from bayesian_network import get_sprinkler_network
from ksd_vi_quantum import KSDVariationalInference 
from utils import plot_training_results, calculate_tvd 

def run_sprinkler_quantum_ksd_experiment():
    print("--- KSD Variational Inference with QuantumBornMachine for Sprinkler P(C,S,R | W=1) ---")
    pytorch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(f"Using PyTorch device: {pytorch_device}")

    pennylane_device_name_for_qbm = "default.qubit" 
    qiskit_digital_twin_backend_obj = None 
    # print(f"Using PennyLane ideal simulator: {pennylane_device_name_for_qbm}")

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
        print("Error: P(Observed) is zero based on the BN. Cannot compute a valid posterior.")
        return

    qbm_num_latent_vars = len(latent_vars_names)
    
    # --- MODIFIED HYPERPARAMETERS FOR EXPERIMENT ---
    qbm_ansatz_layers = 2    # 
    lr_born = 0.002        # Adjusted learning rate (was 0.005, try slightly lower with more params)
    num_epochs_train = 500 # 
    experiment_label = f"L{qbm_ansatz_layers}_LR{lr_born}"
    # --- END MODIFIED HYPERPARAMETERS ---
    
    qbm_conditioning_dim = 0 

    ksd_vi_model = KSDVariationalInference(
        bayesian_network=sprinkler_bn,
        latent_vars_names=latent_vars_names,
        observed_vars_names=observed_vars_names,
        qbm_num_latent_vars=qbm_num_latent_vars,
        qbm_ansatz_layers=qbm_ansatz_layers,
        qbm_conditioning_dim=qbm_conditioning_dim,
        qbm_pennylane_device_name=pennylane_device_name_for_qbm,
        base_kernel_length_scale=1.0,
        pytorch_device=pytorch_device 
    )
    
    print(f"\nINFO: Experiment: {experiment_label}")
    print(f"INFO: QuantumBornMachine initialized with {qbm_num_latent_vars} qubits, {qbm_ansatz_layers} ansatz layer(s).")
    print(f"INFO: Number of trainable parameters in QBM: {sum(p.numel() for p in ksd_vi_model.born_machine.parameters() if p.requires_grad)}")
    print(f"INFO: Learning rate: {lr_born}, Epochs: {num_epochs_train}")

    print("\nStarting KSD training with Quantum Born Machine...")
    
    training_history = ksd_vi_model.train(
        x_observation_dict=x_observation_dict,
        num_epochs=num_epochs_train,
        lr_born_machine=lr_born,
        verbose=True,
        true_posterior_for_tvd=true_posterior_dist
    )

    if training_history:
        plot_title = (f"KSD VI (QBM {experiment_label}) - Sprinkler "
                      f"P({','.join(latent_vars_names)} | W=1)")
        plot_training_results(training_history, title=plot_title)

    print(f"\n--- Comparison of True Posterior and Final Learned Distribution (KSD QBM - {experiment_label}) ---")
    
    qbm_x_cond_input_for_probs = None 
    if ksd_vi_model.num_observed_vars > 0 and ksd_vi_model.born_machine.conditioning_dim > 0:
        x_obs_list_for_qbm_probs = [x_observation_dict[name] for name in observed_vars_names]
        qbm_x_cond_input_for_probs = torch.tensor(x_obs_list_for_qbm_probs, dtype=torch.float32, device=pytorch_device)

    learned_dist_dict = ksd_vi_model.born_machine.get_prob_dict(x_condition=qbm_x_cond_input_for_probs)

    print(f"{'Outcome ' + str(tuple(latent_vars_names)):<20} | {'True Posterior P(z|x)':<25} | {'Learned Q(z|x)':<20} | {'Difference |P-Q|':<20}")
    print("-" * 90)
    
    all_possible_latent_outcomes = sorted(list(true_posterior_dist.keys()))
    for outcome_tuple in all_possible_latent_outcomes:
        true_prob = true_posterior_dist.get(outcome_tuple, 0.0)
        learned_prob = learned_dist_dict.get(outcome_tuple, 0.0) 
        diff = abs(true_prob - learned_prob)
        outcome_str = str(outcome_tuple)
        print(f"{outcome_str:<20} | {true_prob:<25.6f} | {learned_prob:<20.6f} | {diff:<20.6f}")

    final_tvd = calculate_tvd(true_posterior_dist, learned_dist_dict)
    print(f"\nFinal TVD with KSD (Quantum Born Machine - {experiment_label}): {final_tvd:.6f}")

if __name__ == '__main__':
    run_sprinkler_quantum_ksd_experiment()
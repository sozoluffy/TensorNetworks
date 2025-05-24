# quantum_born_machine.py
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class QuantumBornMachine(nn.Module):
    def __init__(self, num_latent_vars, ansatz_layers=1, conditioning_dim=0, 
                 device_name="default.qubit", ansatz_type="hardware_efficient",
                 init_method="small_random"):
        """
        Quantum Born Machine with improved ansatz options and initialization.

        Args:
            num_latent_vars (int): Number of qubits
            ansatz_layers (int): Number of layers in the PQC ansatz
            conditioning_dim (int): Dimension of the conditioning variable x
            device_name (str): PennyLane device name
            ansatz_type (str): Type of ansatz - "hardware_efficient", "all_to_all", or "basic"
            init_method (str): Initialization method - "zero", "small_random", or "random"
        """
        super().__init__()
        self.num_latent_vars = num_latent_vars
        self.conditioning_dim = conditioning_dim
        self.ansatz_type = ansatz_type
        self.ansatz_layers = ansatz_layers

        self.dev = qml.device(device_name, wires=self.num_latent_vars, shots=None)

        # Calculate number of parameters based on ansatz type
        if ansatz_type == "hardware_efficient":
            # Each qubit gets 3 rotation gates per layer
            self.num_ansatz_params = ansatz_layers * 3 * self.num_latent_vars
        elif ansatz_type == "all_to_all":
            # 2 rotations per qubit per layer + extra for entangling
            self.num_ansatz_params = ansatz_layers * 3 * self.num_latent_vars
        else:  # basic
            self.num_ansatz_params = ansatz_layers * 2 * self.num_latent_vars
        
        # Initialize parameters based on chosen method
        if init_method == "zero":
            self.theta = nn.Parameter(torch.zeros(self.num_ansatz_params, dtype=torch.float32))
        elif init_method == "small_random":
            # Small random perturbations around zero
            self.theta = nn.Parameter(0.1 * torch.randn(self.num_ansatz_params, dtype=torch.float32))
        else:  # random
            self.theta = nn.Parameter(torch.rand(self.num_ansatz_params, dtype=torch.float32) * 2 * torch.pi)

        from utils import generate_all_binary_outcomes
        self.all_outcomes_tuples = generate_all_binary_outcomes(self.num_latent_vars)
        if not self.all_outcomes_tuples and self.num_latent_vars > 0:
            raise ValueError("Failed to generate outcome tuples.")
        elif self.num_latent_vars == 0:
            self.all_outcomes_tuples = [()]

        # Define the quantum circuit based on ansatz type
        if ansatz_type == "hardware_efficient":
            @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
            def pqc(weights, x_inputs=None):
                param_idx = 0
                
                # Initial layer of Hadamards for superposition
                for i in range(self.num_latent_vars):
                    qml.Hadamard(wires=i)
                
                for layer_idx in range(ansatz_layers):
                    # Single qubit rotations
                    for i in range(self.num_latent_vars):
                        qml.RX(weights[param_idx], wires=i); param_idx += 1
                        qml.RY(weights[param_idx], wires=i); param_idx += 1
                        qml.RZ(weights[param_idx], wires=i); param_idx += 1
                    
                    # Entangling gates
                    if self.num_latent_vars > 1:
                        # Nearest neighbor
                        for i in range(self.num_latent_vars - 1):
                            qml.CNOT(wires=[i, i + 1])
                        # Wrap around for ring connectivity
                        if self.num_latent_vars > 2:
                            qml.CNOT(wires=[self.num_latent_vars - 1, 0])
                        
                        # Additional entanglement for even layers
                        if layer_idx % 2 == 0 and self.num_latent_vars > 2:
                            for i in range(0, self.num_latent_vars - 2, 2):
                                qml.CZ(wires=[i, i + 2])
                
                return qml.probs(wires=range(self.num_latent_vars))
                
        elif ansatz_type == "all_to_all":
            @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
            def pqc(weights, x_inputs=None):
                param_idx = 0
                
                # Initial superposition
                for i in range(self.num_latent_vars):
                    qml.Hadamard(wires=i)
                
                for layer_idx in range(ansatz_layers):
                    # Single qubit rotations
                    for i in range(self.num_latent_vars):
                        qml.RX(weights[param_idx], wires=i); param_idx += 1
                        qml.RY(weights[param_idx], wires=i); param_idx += 1
                        qml.RZ(weights[param_idx], wires=i); param_idx += 1
                    
                    # All-to-all entanglement
                    if self.num_latent_vars > 1:
                        for i in range(self.num_latent_vars):
                            for j in range(i + 1, self.num_latent_vars):
                                qml.CZ(wires=[i, j])
                
                return qml.probs(wires=range(self.num_latent_vars))
                
        else:  # basic ansatz (original)
            @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
            def pqc(weights, x_inputs=None):
                param_idx = 0
                for layer_idx in range(ansatz_layers):
                    for i in range(self.num_latent_vars):
                        qml.RY(weights[param_idx], wires=i); param_idx += 1
                        qml.RZ(weights[param_idx], wires=i); param_idx += 1
                    
                    if self.num_latent_vars > 1:
                        for i in range(self.num_latent_vars - 1):
                            qml.CNOT(wires=[i, i + 1])
                        if self.num_latent_vars > 2:
                            qml.CNOT(wires=[self.num_latent_vars - 1, 0])
                
                return qml.probs(wires=range(self.num_latent_vars))
            
        self.pqc = pqc

    def get_probabilities(self, x_condition=None):
        """Computes the probability distribution q_theta(z|x) over all 2^N states."""
        if self.conditioning_dim > 0 and x_condition is not None:
            print("Warning: Conditioning with x_condition not fully implemented in PQC ansatz yet.")
            
        return self.pqc(weights=self.theta)

    def get_prob_dict(self, x_condition=None):
        """Returns the probability distribution as a dictionary."""
        probs_tensor = self.get_probabilities(x_condition=x_condition)
        
        if probs_tensor.shape[0] != len(self.all_outcomes_tuples):
            raise ValueError(f"Mismatch between probability tensor shape and number of outcomes")

        probs_list = probs_tensor.detach().cpu().tolist()
        
        prob_dict = {}
        for i, outcome_tuple in enumerate(self.all_outcomes_tuples):
            prob_dict[outcome_tuple] = probs_list[i]
        return prob_dict

    def sample(self, num_samples_to_draw, x_condition=None):
        """Samples from the Born machine's distribution."""
        if self.num_latent_vars == 0:
            return torch.empty(num_samples_to_draw, 0, dtype=torch.float32, device=self.theta.device)

        probs = self.get_probabilities(x_condition=x_condition).detach().cpu()
        
        if probs.ndim > 1:
            if probs.shape[0] == 1 and probs.ndim == 2:
                probs = probs.squeeze(0)
            else:
                raise ValueError(f"Probabilities for sampling should be 1D, but got shape {probs.shape}")
        
        # Ensure probs sum to 1
        probs = probs / torch.sum(probs)

        sampled_indices = torch.multinomial(probs, num_samples_to_draw, replacement=True)
        
        samples_list = []
        for idx in sampled_indices:
            samples_list.append(torch.tensor(self.all_outcomes_tuples[idx.item()], dtype=torch.float32))
        
        if not samples_list:
            return torch.empty(0, self.num_latent_vars, dtype=torch.float32, device=self.theta.device)
            
        return torch.stack(samples_list).to(self.theta.device)

    def get_log_q_z_x(self, z_samples_batch, x_condition=None):
        """Calculates log q_theta(z | x) for given samples."""
        if self.num_latent_vars == 0:
            if z_samples_batch.shape[0] == 0:
                return torch.empty(0, device=self.theta.device)
            return torch.zeros(z_samples_batch.shape[0], device=self.theta.device)

        probs_all_states = self.get_probabilities(x_condition=x_condition)
        log_probs_all_states = torch.log(probs_all_states.clamp(min=1e-9))

        selected_log_probs = []
        for z_sample_tensor in z_samples_batch:
            z_sample_tuple = tuple(z_sample_tensor.cpu().long().tolist())
            try:
                outcome_idx = self.all_outcomes_tuples.index(z_sample_tuple)
                selected_log_probs.append(log_probs_all_states[outcome_idx])
            except ValueError:
                raise ValueError(f"Sample {z_sample_tuple} is not a valid outcome")
        
        if not selected_log_probs:
            return torch.empty(0, device=self.theta.device)
        return torch.stack(selected_log_probs)
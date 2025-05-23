# quantum_born_machine.py
import torch
import torch.nn as nn
import pennylane as qml
# from pennylane import numpy as pnp # PennyLane's NumPy, usually not needed when interface="torch"

# Assuming generate_all_binary_outcomes is in your utils.py
# from utils import generate_all_binary_outcomes # Will be imported in __main__ if fallback needed

class QuantumBornMachine(nn.Module):
    def __init__(self, num_latent_vars, ansatz_layers=1, conditioning_dim=0, device_name="default.qubit"):
        """
        Quantum Born Machine implemented using PennyLane with a PyTorch interface.

        Args:
            num_latent_vars (int): Number of qubits, corresponding to the number of latent variables.
            ansatz_layers (int): Number of layers in the PQC ansatz.
            conditioning_dim (int): Dimension of the conditioning variable x. (Not fully implemented in ansatz yet)
            device_name (str): PennyLane device name (e.g., "default.qubit", "lightning.qubit").
                               Later, can be "qiskit.aer" for your digital twins.
        """
        super().__init__()
        self.num_latent_vars = num_latent_vars
        self.conditioning_dim = conditioning_dim # Placeholder for future use in ansatz

        # Define the PennyLane quantum device. shots=None for exact probabilities.
        self.dev = qml.device(device_name, wires=self.num_latent_vars, shots=None)

        # Determine the number of parameters for the ansatz
        # Example: Hardware-efficient ansatz inspired by Fig. 5 of PhysRevApplied.16.044057
        # Each layer has RY, RZ per qubit, then a ring of CNOTs.
        # So, 2 parameters per qubit per layer.
        self.num_ansatz_params = ansatz_layers * 2 * self.num_latent_vars
        
        # PyTorch parameters for the PQC weights (theta)
        # Initialize with random values, e.g., from a uniform distribution
        # Ensure theta is float32 by default, PennyLane will handle casting if device uses float64
        self.theta = nn.Parameter(torch.rand(self.num_ansatz_params, dtype=torch.float32) * 2 * torch.pi)


        # Define the QNode: this binds the PQC to the device and makes it differentiable
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def pqc(weights, x_inputs=None): # x_inputs for potential conditioning
            # Example PQC Ansatz (Hardware-efficient style)
            # `weights` will be self.theta
            # `x_inputs` is a placeholder if conditioning data needs to be passed directly
            
            param_idx = 0
            for layer_idx in range(ansatz_layers):
                # Layer of single-qubit rotations
                for i in range(self.num_latent_vars):
                    qml.RY(weights[param_idx], wires=i); param_idx += 1
                    qml.RZ(weights[param_idx], wires=i); param_idx += 1
                
                # Layer of entangling gates (e.g., CNOTs in a ring or ladder)
                if self.num_latent_vars > 1: # Add entanglers if more than 1 qubit
                    for i in range(self.num_latent_vars -1):
                        qml.CNOT(wires=[i, i + 1])
                    if self.num_latent_vars > 2 : # To make it a ring for >2 qubits for better entanglement
                         qml.CNOT(wires=[self.num_latent_vars - 1, 0])
            
            # Return probabilities of all computational basis states
            return qml.probs(wires=range(self.num_latent_vars))
            
        self.pqc = pqc
        
        # For converting indices to outcome tuples and vice-versa
        # This needs to be imported or defined. Assuming it's available.
        from utils import generate_all_binary_outcomes # Placed here for clarity
        self.all_outcomes_tuples = generate_all_binary_outcomes(self.num_latent_vars)
        if not self.all_outcomes_tuples and self.num_latent_vars > 0: 
            raise ValueError("Failed to generate outcome tuples. Check num_latent_vars.")
        elif self.num_latent_vars == 0: 
            self.all_outcomes_tuples = [()]


    def get_probabilities(self, x_condition=None):
        """
        Computes the probability distribution q_theta(z|x) over all 2^N states.
        Returns a torch.Tensor of shape (2**num_latent_vars,).
        """
        if self.conditioning_dim > 0 and x_condition is not None:
            print("Warning: Conditioning with x_condition not fully implemented in PQC ansatz yet.")
            pass 
            
        # Ensure weights passed to pqc match the dtype expected by PennyLane qnode if necessary,
        # though PennyLane usually handles torch.float32 inputs well.
        return self.pqc(weights=self.theta)

    def sample(self, num_samples_to_draw, x_condition=None):
        """
        Samples from the Born machine's distribution q_theta(z|x).

        Args:
            num_samples_to_draw (int): Number of samples to generate.
            x_condition (torch.Tensor, optional): Conditioning variable.

        Returns:
            torch.Tensor: Shape (num_samples_to_draw, num_latent_vars)
        """
        if self.num_latent_vars == 0:
            return torch.empty(num_samples_to_draw, 0, dtype=torch.float32, device=self.theta.device)

        # .detach() as sampling itself is not end-to-end differentiable for REINFORCE policy
        # The log_prob of the action is what's differentiated.
        # Ensure probs are on CPU for multinomial if it has issues with CUDA tensors without explicit handling
        probs = self.get_probabilities(x_condition=x_condition).detach().cpu() 
        
        if probs.ndim > 1:
            if probs.shape[0] == 1 and probs.ndim == 2: probs = probs.squeeze(0)
            else: raise ValueError(f"Probabilities for sampling should be 1D, but got shape {probs.shape}")
        if probs.shape[0] == 0 and self.num_latent_vars > 0 : # Check for empty probs only if expecting >0 outcomes
             raise ValueError("Probability vector is empty.")


        sampled_indices = torch.multinomial(probs, num_samples_to_draw, replacement=True)
        
        samples_list = []
        for idx in sampled_indices:
            samples_list.append(torch.tensor(self.all_outcomes_tuples[idx.item()], dtype=torch.float32))
        
        if not samples_list: 
            return torch.empty(0, self.num_latent_vars, dtype=torch.float32, device=self.theta.device)
            
        return torch.stack(samples_list).to(self.theta.device)

    def get_log_q_z_x(self, z_samples_batch, x_condition=None):
        """
        Calculates log q_theta(z | x) for given samples z_samples_batch.
        This is differentiable w.r.t. self.theta due to PennyLane's torch interface.

        Args:
            z_samples_batch (torch.Tensor): Samples, shape (batch_size, num_latent_vars).
            x_condition (torch.Tensor, optional): Conditioning variables.

        Returns:
            torch.Tensor: log probabilities, shape (batch_size,).
        """
        if self.num_latent_vars == 0:
            if z_samples_batch.shape[0] == 0: return torch.empty(0, device=self.theta.device)
            return torch.zeros(z_samples_batch.shape[0], device=self.theta.device)

        probs_all_states = self.get_probabilities(x_condition=x_condition)
        log_probs_all_states = torch.log(probs_all_states + 1e-9) 

        selected_log_probs = []
        for z_sample_tensor in z_samples_batch:
            z_sample_tuple = tuple(z_sample_tensor.cpu().long().tolist())
            try:
                outcome_idx = self.all_outcomes_tuples.index(z_sample_tuple)
                selected_log_probs.append(log_probs_all_states[outcome_idx])
            except ValueError:
                raise ValueError(f"Sample {z_sample_tuple} is not a valid outcome for this {self.num_latent_vars}-qubit Born machine.")
        
        if not selected_log_probs: 
            return torch.empty(0, device=self.theta.device)
        return torch.stack(selected_log_probs)

if __name__ == '__main__':
    # Ensure generate_all_binary_outcomes is accessible for the test
    try:
        from utils import generate_all_binary_outcomes
    except ImportError:
        def generate_all_binary_outcomes(num_vars): # Fallback
            if num_vars == 0: return [()]
            outcomes = []
            for i in range(2**num_vars):
                binary_representation = bin(i)[2:].zfill(num_vars)
                outcomes.append(tuple(map(int, list(binary_representation))))
            return outcomes
        print("Used fallback generate_all_binary_outcomes for testing.")


    num_qubits = 3
    layers = 1
    qbm = QuantumBornMachine(num_latent_vars=num_qubits, ansatz_layers=layers)
    print(f"QuantumBornMachine initialized with {num_qubits} qubits and {qbm.num_ansatz_params} parameters.")
    print("Initial parameters (theta):", qbm.theta)

    print("\n--- Testing get_probabilities ---")
    probs = qbm.get_probabilities()
    print("Probabilities shape:", probs.shape)
    print("Probabilities sum:", torch.sum(probs))
    print("Probabilities vector (first few):", probs[:min(5, len(probs))].tolist())
    # Corrected assertion: make the comparison tensor dtype torch.float64
    assert torch.isclose(torch.sum(probs), torch.tensor(1.0, dtype=probs.dtype)), "Probabilities should sum to 1"


    print("\n--- Testing sample ---")
    num_s = 5
    samples = qbm.sample(num_samples_to_draw=num_s)
    print(f"Sampled {num_s} states (shape {samples.shape}):\n", samples)
    assert samples.shape == (num_s, num_qubits), "Samples shape mismatch"

    print("\n--- Testing get_log_q_z_x and gradients ---")
    if samples.shape[0] > 0: 
        test_z_batch = samples.to(qbm.theta.device) 
        
        log_probs_of_samples = qbm.get_log_q_z_x(test_z_batch)
        print("Log_probs of sampled states:", log_probs_of_samples)
        
        # Ensure dummy_loss is of the same dtype as theta for backward pass
        dummy_loss = -log_probs_of_samples.mean().type_as(qbm.theta)
        print("Dummy loss:", dummy_loss.item())

        qbm.theta.grad = None 
        dummy_loss.backward()
        
        if qbm.theta.grad is not None:
            print("Gradients for theta computed successfully!")
            print("Gradient (first few params):", qbm.theta.grad[:min(5, len(qbm.theta.grad))].tolist())
        else:
            print("Gradients for theta are None. Something went wrong with diff.")
    else:
        print("No samples drawn, skipping gradient test.")

    print("\nQuantumBornMachine basic tests complete.")
# Quantum-Inspired Variational Inference in Python

This project provides a Python-based classical simulation framework for variational inference (VI) techniques inspired by the paper "Variational Inference with a Quantum Computer" (PhysRevApplied.16.044057). It focuses on implementing the adversarial and Kernelized Stein Discrepancy (KSD) methods for approximate inference in Bayesian networks.

**Disclaimer:** This is a classical simulation. It does *not* implement actual quantum circuits or aim to demonstrate quantum advantage. Instead, it explores the algorithmic aspects of using Born-machine-like parameterized distributions for VI within a classical machine learning context.

## Project Status

* **Adversarial Variational Inference (KL-based):** Successfully implemented for discrete Bayesian Networks.
    * The method utilizes a classical analogue of a Born machine (a parameterized probability distribution) and an adversarial classifier.
    * Training employs the REINFORCE algorithm for the Born machine's gradient estimation, which was crucial for effective learning.
    * Demonstrated promising results on the "Sprinkler" Bayesian network, achieving significant reduction in Total Variation Distance (TVD) between the learned and true posterior distributions.
* **Kernelized Stein Discrepancy (KSD) Variational Inference:** Successfully implemented for discrete Bayesian Networks.
    * Uses an exact sum computation for the KSD objective (feasible for small state spaces like the Sprinkler network), avoiding sampling for the KSD term itself.
    * Demonstrated excellent results on the "Sprinkler" Bayesian network, achieving very low TVD (e.g., ~0.0004) and a close match to the true posterior.
* **Next Major Step:** Exploring quantum implementations/simulations.

## Project Goals

* Implement a flexible way to define and sample from discrete Bayesian Networks.
* Simulate a "Born Machine" classically as a parameterized probability distribution.
* Implement the adversarial VI training loop, including a neural network classifier and REINFORCE gradient estimation.
* Implement the Kernelized Stein Discrepancy (KSD) VI method using an exact kernel sum for small state spaces.
* Provide example usage for simple Bayesian networks like the "Sprinkler" network for both methods.
* (Future) Explore quantum circuit implementations for the Born machine.

## Structure

* `bayesian_network.py`: Defines Bayesian Network structures, conditional probability tables (CPTs), and sampling.
* `born_machine_classical_sim.py`: Implements a classical, parameterized probability distribution that serves as an analogue to the quantum Born machine.
* `classifier_pytorch.py`: A simple neural network classifier (e.g., MLP) implemented using PyTorch, used in the adversarial VI method.
* `adversarial_vi.py`: Contains the logic for the adversarial training procedure.
* `stein_utils.py`: Helper functions specific to the KSD method (score function, Stein kernel components).
* `ksd_vi.py`: Contains the logic for the KSD-based VI training procedure.
* `utils.py`: General utility functions (e.g., for calculating Total Variation Distance (TVD), plotting, generating binary outcomes).
* `run_sprinkler_adversarial.py`: Example script to demonstrate adversarial VI on the Sprinkler network.
* `run_sprinkler_ksd.py`: Example script to demonstrate KSD VI on the Sprinkler network.
* `requirements.txt`: Lists necessary Python packages.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sozoluffy/TensorNetworks.git](https://github.com/sozoluffy/TensorNetworks.git)
    cd TensorNetworks
    ```

2.  **Create a virtual environment and install dependencies:**
    * **Using `venv`:**
        ```bash
        python3 -m venv venv  # Or python -m venv venv on Windows
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        pip install -r requirements.txt
        ```
    * **Using `conda`:**
        ```bash
        conda create --name qvi_env python=3.9 # Or your preferred python version
        conda activate qvi_env
        pip install -r requirements.txt
        # Alternatively, for a more conda-centric approach, create an environment.yml
        ```

3.  **Run an Example:**
    * For Adversarial VI:
        ```bash
        python run_sprinkler_adversarial.py
        ```
    * For KSD VI:
        ```bash
        python run_sprinkler_ksd.py
        ```

## Key Concepts from the Paper Implemented (Classically)

* **Variational Inference (VI):** Approximating a complex target probability distribution (the posterior) with a simpler, parameterized distribution (the variational distribution) by optimizing its parameters.
* **Born Machine (Classical Analogue):** A quantum circuit would produce samples based on Born rule. Our classical analogue uses a neural network to output a categorical distribution.
* **Adversarial Objective (KL Divergence based):** A classifier distinguishes samples from the Born machine and a prior. The Born machine is trained to fool this classifier.
* **REINFORCE Algorithm:** Used for gradient estimation for the Born machine in the adversarial setup due to discrete sampling.
* **Kernelized Stein Discrepancy (KSD):** An alternative VI objective measuring discrepancy using a kernel function and the Stein operator. Our implementation uses an exact sum for $K_x(\theta)$ for small problems.

## Potential Next Steps for Refinement & Quantum Exploration

1.  **Quantum Implementation of Born Machine:**
    * Replace `ClassicalBornMachine` with a Parameterized Quantum Circuit (PQC) using a library like Qiskit, Cirq, or PennyLane.
    * Implement gradient estimation for PQCs (e.g., parameter-shift rule).
    * Adapt VI algorithms to use the PQC, its sampling methods, and gradient calculations.
    * Explore ansatz design and address quantum-specific challenges (barren plateaus, noise if using real hardware).
2.  **Saving the Best Model:** Modify training loops to save model parameters achieving the lowest TVD.
3.  **Learning Rate Schedule & Advanced Hyperparameter Tuning:** Implement learning rate decay and further tune parameters for both VI methods.
4.  **REINFORCE with a Baseline (for Adversarial VI):** Implement a baseline to reduce variance for more stable training.
5.  **Sample-based KSD:** For larger state spaces where the exact sum for $K_x(\theta)$ is infeasible, implement the sample-based estimation of the KSD objective and its gradient (potentially requiring REINFORCE-like ideas for $\nabla_\theta K_x(\theta)$ if not using parameter-shift on a PQC directly).
6.  **More Complex Bayesian Networks:** Test with larger networks (e.g., "Lung Cancer" network).
7.  **Amortization:** Fully implement and test amortization where the Born machine conditions on various observations `x` from a dataset.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

(Consider adding a license, e.g., MIT License. If you do, create a `LICENSE` file.)
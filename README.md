# Quantum-Inspired Variational Inference in Python

This project provides a Python-based classical simulation framework for variational inference (VI) techniques inspired by the paper "Variational Inference with a Quantum Computer" (PhysRevApplied.16.044057). It focuses on implementing the adversarial and (eventually) kernelized Stein discrepancy (KSD) methods for approximate inference in Bayesian networks.

**Disclaimer:** This is a classical simulation. It does *not* implement actual quantum circuits or aim to demonstrate quantum advantage. Instead, it explores the algorithmic aspects of using Born-machine-like parameterized distributions for VI within a classical machine learning context.

## Project Goals

* Implement a flexible way to define and sample from discrete Bayesian Networks.
* Simulate a "Born Machine" classically as a parameterized probability distribution.
* Implement the adversarial VI training loop, including a neural network classifier.
* (Future) Implement the Kernelized Stein Discrepancy (KSD) VI method.
* Provide example usage for simple Bayesian networks like the "Sprinkler" network.

## Structure

* `bayesian_network.py`: Defines Bayesian Network structures, conditional probability tables (CPTs), and sampling.
* `born_machine_classical_sim.py`: Implements a classical, parameterized probability distribution that serves as an analogue to the quantum Born machine.
* `classifier_pytorch.py`: A simple neural network classifier (e.g., MLP) implemented using PyTorch, used in the adversarial VI method.
* `adversarial_vi.py`: Contains the logic for the adversarial training procedure, optimizing the Born machine and the classifier.
* `ksd_vi.py`: (Placeholder/Future) Will contain the logic for the KSD-based VI.
* `utils.py`: Utility functions (e.g., for calculating Total Variation Distance (TVD), plotting).
* `run_sprinkler_adversarial.py`: Example script to demonstrate adversarial VI on the Sprinkler network.
* `requirements.txt`: Lists necessary Python packages.

## Getting Started

1.  **Clone the repository (once you create it on GitHub):**
    ```bash
    git clone <your-repo-url>
    cd quantum-inspired-vi
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run an example:**
    ```bash
    python run_sprinkler_adversarial.py
    ```

## Key Concepts from the Paper

* **Variational Inference (VI):** Approximating a complex target probability distribution (the posterior) with a simpler, parameterized distribution (the variational distribution) by optimizing its parameters.
* **Born Machine:** A quantum circuit that produces classical data samples according to probabilities defined by the Born rule ($q(z) = |\langle z|\psi(\theta)\rangle|^2$). In this project, it's simulated classically.
* **Adversarial Objective (KL Divergence based):** The VI objective is related to the Kullback-Leibler (KL) divergence. The paper uses a method where a classifier is trained to distinguish samples from the Born machine and a prior distribution. The Born machine is then trained to fool this classifier, effectively minimizing a proxy for the KL divergence to the target posterior.
* **Kernelized Stein Discrepancy (KSD):** An alternative VI objective that measures the discrepancy between distributions using a kernel function and the Stein operator. It doesn't require a separate adversarial network.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

(Specify your preferred license, e.g., MIT License)

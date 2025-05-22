# Quantum-Inspired Variational Inference in Python

This project provides a Python-based classical simulation framework for variational inference (VI) techniques inspired by the paper "Variational Inference with a Quantum Computer" (PhysRevApplied.16.044057). It focuses on implementing the adversarial method for approximate inference in Bayesian networks, with future plans for the kernelized Stein discrepancy (KSD) method.

**Disclaimer:** This is a classical simulation. It does *not* implement actual quantum circuits or aim to demonstrate quantum advantage. Instead, it explores the algorithmic aspects of using Born-machine-like parameterized distributions for VI within a classical machine learning context.

## Project Status

* **Adversarial Variational Inference:** Successfully implemented for discrete Bayesian Networks.
    * The method utilizes a classical analogue of a Born machine (a parameterized probability distribution) and an adversarial classifier.
    * Training employs the REINFORCE algorithm for the Born machine's gradient estimation, which proved crucial for effective learning.
    * Demonstrated promising results on the "Sprinkler" Bayesian network, achieving significant reduction in Total Variation Distance (TVD) between the learned and true posterior distributions (e.g., P(Cloudy, Sprinkler, Rain | Grass Wet = true)).
* **Kernelized Stein Discrepancy (KSD) VI:** Planned for future implementation.

## Project Goals

* Implement a flexible way to define and sample from discrete Bayesian Networks.
* Simulate a "Born Machine" classically as a parameterized probability distribution.
* Implement the adversarial VI training loop, including a neural network classifier and REINFORCE gradient estimation.
* (Future) Implement the Kernelized Stein Discrepancy (KSD) VI method.
* Provide example usage for simple Bayesian networks like the "Sprinkler" network.

## Structure

* `bayesian_network.py`: Defines Bayesian Network structures, conditional probability tables (CPTs), and sampling.
* `born_machine_classical_sim.py`: Implements a classical, parameterized probability distribution that serves as an analogue to the quantum Born machine.
* `classifier_pytorch.py`: A simple neural network classifier (e.g., MLP) implemented using PyTorch, used in the adversarial VI method.
* `adversarial_vi.py`: Contains the logic for the adversarial training procedure, optimizing the Born machine and the classifier.
* `utils.py`: Utility functions (e.g., for calculating Total Variation Distance (TVD), plotting).
* `run_sprinkler_adversarial.py`: Example script to demonstrate adversarial VI on the Sprinkler network.
* `requirements.txt`: Lists necessary Python packages.
* (Future) `ksd_vi.py`: Will contain the logic for the KSD-based VI.

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

3.  **Run the Sprinkler Network Example:**
    ```bash
    python run_sprinkler_adversarial.py
    ```

## Key Concepts from the Paper Implemented

* **Variational Inference (VI):** Approximating a complex target probability distribution (the posterior) with a simpler, parameterized distribution (the variational distribution) by optimizing its parameters.
* **Born Machine (Classical Analogue):** A quantum circuit that produces classical data samples according to probabilities defined by the Born rule ($q(z) = |\langle z|\psi(\theta)\rangle|^2$). In this project, it's simulated classically as a neural network outputting a categorical distribution.
* **Adversarial Objective (KL Divergence based):** The VI objective is related to the Kullback-Leibler (KL) divergence. The paper uses a method where a classifier is trained to distinguish samples from the Born machine and a prior distribution. The Born machine is then trained to fool this classifier, effectively minimizing a proxy for the KL divergence to the target posterior.
* **REINFORCE Algorithm:** Used for estimating the gradient of the Born machine's objective, as the sampling from the Born machine is a discrete, non-differentiable operation.

## Potential Next Steps for Refinement

1.  **Saving the Best Model:** Modify the training loop to keep track of and save the model parameters (Born machine and classifier) that achieve the lowest TVD during training.
2.  **Learning Rate Schedule:** Implement learning rate decay or schedulers for more stable convergence in later stages of training.
3.  **Hyperparameter Tuning:** Further tune learning rates, batch sizes, network architectures, and the ratio of classifier to Born machine training steps.
4.  **REINFORCE with a Baseline:** Implement a baseline (e.g., moving average of rewards) in the REINFORCE algorithm to reduce gradient variance and potentially stabilize training.
5.  **Kernelized Stein Discrepancy (KSD):** Implement the second VI method proposed in the paper.
6.  **More Complex Bayesian Networks:** Test and adapt the framework for larger networks (e.g., "Lung Cancer" network from the paper).
7.  **Amortization:** Extend the `ClassicalBornMachine` and training framework to support amortized inference where the Born machine conditions on different observations `x` from a dataset $p_D(x)$.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

(Consider adding a license, e.g., MIT License. If you do, create a `LICENSE` file.)
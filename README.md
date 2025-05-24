# Quantum-Inspired Variational Inference in Python

A Python implementation of variational inference techniques inspired by the paper "Variational Inference with a Quantum Computer" (PhysRevApplied.16.044057). This project provides both classical simulations and quantum implementations of Born machine-based variational inference for discrete Bayesian networks.

## Overview

This project implements two main variational inference approaches:

1. **Adversarial Variational Inference (KL-based)**: Uses an adversarial classifier to distinguish between samples from the Born machine and prior distribution
2. **Kernelized Stein Discrepancy (KSD) Variational Inference**: Minimizes the KSD between the variational and target distributions

Both methods are implemented with:
- **Classical Born Machines**: Neural network-based parameterized distributions
- **Quantum Born Machines**: Parameterized quantum circuits using PennyLane

## Features

- ✅ Flexible Bayesian Network framework for discrete variables
- ✅ Classical Born machine with neural network parameterization
- ✅ Quantum Born machine with multiple ansatz options
- ✅ Adversarial VI with REINFORCE gradient estimation
- ✅ KSD VI with exact computation for small state spaces
- ✅ Comprehensive training utilities with stability enhancements
- ✅ Detailed visualization and analysis tools

## Project Structure

```
.
├── bayesian_network.py           # Bayesian Network implementation
├── born_machine_classical_sim.py # Classical Born machine
├── quantum_born_machine.py       # Quantum Born machine (PennyLane)
├── classifier_pytorch.py         # Neural network classifier for adversarial VI
├── adversarial_vi.py            # Adversarial VI training logic
├── ksd_vi.py                    # Classical KSD VI implementation
├── ksd_vi_quantum.py            # Quantum KSD VI implementation
├── stein_utils.py               # KSD helper functions
├── utils.py                     # General utilities (TVD, plotting, etc.)
├── run_sprinkler_adversarial.py # Example: Adversarial VI on Sprinkler network
├── run_sprinkler_ksd.py         # Example: Classical KSD VI on Sprinkler network
├── run_sprinkler_quantum_ksd.py # Example: Quantum KSD VI on Sprinkler network
└── requirements.txt             # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sozoluffy/TensorNetworks.git
cd TensorNetworks
```

2. Create a virtual environment:

**Using venv:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create --name qvi_env python=3.9
conda activate qvi_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For quantum implementations, also install PennyLane:
```bash
pip install pennylane
```

## Quick Start

### Running Adversarial VI

```bash
python run_sprinkler_adversarial.py
```

This demonstrates adversarial variational inference on the Sprinkler Bayesian network, learning P(C,S,R | W=1).

### Running Classical KSD VI

```bash
python run_sprinkler_ksd.py
```

This runs KSD-based variational inference using a classical Born machine.

### Running Quantum KSD VI

```bash
python run_sprinkler_quantum_ksd.py
```

This demonstrates KSD VI using a quantum Born machine implemented with PennyLane.

## Key Concepts

### Bayesian Networks
The framework supports discrete Bayesian networks with binary variables. Example networks include:
- **Sprinkler Network**: Classic 4-variable network (Cloudy, Sprinkler, Rain, Grass Wet)
- Custom networks can be easily defined using conditional probability tables (CPTs)

### Born Machines
- **Classical**: Uses neural networks to parameterize probability distributions
- **Quantum**: Uses parameterized quantum circuits (PQCs) with measurement in computational basis

### Variational Inference Methods

#### Adversarial VI
- Objective: min_θ KL[q_θ(z|x) || p(z|x)]
- Uses adversarial classifier to estimate log density ratio
- REINFORCE algorithm for gradient estimation
- Features: baseline variance reduction, gradient clipping, learning rate scheduling

#### KSD VI
- Objective: min_θ KSD[q_θ(z|x), p(z|x)]
- Direct minimization of kernelized Stein discrepancy
- Exact computation for small state spaces
- Gradient-based optimization with parameter-shift rule for quantum circuits

## Detailed Usage

### Creating a Bayesian Network

```python
from bayesian_network import BayesianNetwork

# Create a simple A → B network
bn = BayesianNetwork()
bn.add_node('A', cpt={(): {0: 0.7, 1: 0.3}})
bn.add_node('B', cpt={
    (0,): {0: 0.9, 1: 0.1},  # P(B|A=0)
    (1,): {0: 0.2, 1: 0.8}   # P(B|A=1)
}, parent_names=['A'])
```

### Running Adversarial VI

```python
from adversarial_vi import AdversarialVariationalInference
from bayesian_network import get_sprinkler_network

# Setup
bn = get_sprinkler_network()
latent_vars = ['C', 'S', 'R']
observed_vars = ['W']
observation = {'W': 1}

# Configure Born machine and classifier
born_config = {
    'use_logits': True,
    'conditioning_dim': len(observed_vars),
    'init_method': 'uniform'
}
classifier_config = {
    'hidden_dims': [32, 16],
    'use_batch_norm': False
}

# Initialize and train
model = AdversarialVariationalInference(
    bn, latent_vars, observed_vars,
    born_config, classifier_config
)

history = model.train(
    observation,
    num_epochs=1500,
    batch_size=100,
    lr_born_machine=0.003,
    lr_classifier=0.03,
    k_classifier_steps=5,
    k_born_steps=1
)
```

### Running KSD VI

```python
from ksd_vi import KSDVariationalInference

# Similar setup as above...

model = KSDVariationalInference(
    bn, latent_vars, observed_vars,
    born_config,
    base_kernel_length_scale=1.0
)

history = model.train(
    observation,
    num_epochs=2000,
    lr_born_machine=0.003,
    entropy_weight=0.001,
    patience=200
)
```

## Training Parameters

### Common Parameters
- `num_epochs`: Number of training iterations
- `lr_born_machine`: Learning rate for Born machine parameters
- `use_lr_scheduler`: Enable cosine annealing (recommended)
- `gradient_clip_norm`: Maximum gradient norm for clipping
- `optimizer_type`: "adam" or "sgd"

### Adversarial VI Specific
- `batch_size`: Number of samples per batch
- `lr_classifier`: Learning rate for adversarial classifier
- `k_classifier_steps`: Classifier updates per Born machine update
- `k_born_steps`: Born machine updates per iteration
- `baseline_decay`: Exponential decay for REINFORCE baseline

### KSD VI Specific
- `entropy_weight`: Regularization weight for entropy bonus
- `patience`: Early stopping patience (epochs without improvement)

## Quantum Implementation Details

### Ansatz Options
1. **Hardware Efficient**: Single-qubit rotations + nearest-neighbor entanglement
2. **All-to-All**: Single-qubit rotations + all-to-all entanglement
3. **Basic**: Simple RY-RZ rotations with linear entanglement

### Initialization Methods
- `"zero"`: All parameters initialized to zero
- `"small_random"`: Small random perturbations (recommended)
- `"random"`: Random initialization in [0, 2π]

## Evaluation Metrics

- **Total Variation Distance (TVD)**: Primary metric for distribution comparison
- **KSD Loss**: Kernelized Stein discrepancy value
- **Gradient Norms**: For monitoring training stability
- **Entropy**: Distribution entropy (for KSD regularization)

## Visualization

The training scripts automatically generate plots showing:
- Loss curves over epochs
- TVD evolution
- Gradient norms
- Probability distribution comparisons
- Stability metrics

## Contributing

Contributions are welcome! Areas for improvement include:
- Extended to continuous variables
- More sophisticated quantum ansätze
- Amortized inference implementations
- Support for larger Bayesian networks
- Hardware quantum device integration

## Citation

If you use this code in your research, please cite:
```
@article{PhysRevApplied.16.044057,
  title = {Variational inference with a quantum computer},
  author = {Marcello Benedetti, Brian Coyle, Mattia Fiorentini, Michael Lubasch, Matthias Rosenkranz},
  journal = {Phys. Rev. Applied},
  volume = {16},
  pages = {044057},
  year = {2021}
}
```

## License

[Add your license here - e.g., MIT, Apache 2.0]

## Acknowledgments

This implementation is based on the theoretical framework presented in "Variational Inference with a Quantum Computer" and serves as an educational resource for understanding quantum-inspired machine learning algorithms.
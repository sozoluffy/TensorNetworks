# utils.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def calculate_tvd(p_true, p_approx):
    """
    Calculates the Total Variation Distance (TVD) between two discrete probability distributions.

    Args:
        p_true (dict or np.array): The true probability distribution.
                                   If dict, keys are outcomes, values are probabilities.
                                   If np.array, indices are outcomes.
        p_approx (dict or np.array): The approximate probability distribution.
                                     Format should match p_true.

    Returns:
        float: The Total Variation Distance.
    """
    if isinstance(p_true, dict) and isinstance(p_approx, dict):
        all_outcomes = set(p_true.keys()) | set(p_approx.keys())
        tvd = 0.0
        for outcome in all_outcomes:
            prob_true = p_true.get(outcome, 0.0)
            prob_approx = p_approx.get(outcome, 0.0)
            tvd += np.abs(prob_true - prob_approx)
        return 0.5 * tvd
    elif isinstance(p_true, np.ndarray) and isinstance(p_approx, np.ndarray):
        if p_true.shape != p_approx.shape:
            # This is a simplification. For full generality, one might need to align
            # distributions over a common support. For this project, we assume
            # they are defined over the same ordered set of outcomes.
            raise ValueError("Probability arrays must have the same shape for simple TVD calculation.")
        return 0.5 * np.sum(np.abs(p_true - p_approx))
    else:
        raise TypeError("Inputs p_true and p_approx must be both dicts or both np.arrays.")

def plot_training_results(results_dict, title="Training Results"):
    """
    Plots various metrics recorded during training.

    Args:
        results_dict (dict): A dictionary where keys are metric names (e.g., 'tvd', 'loss_born', 'loss_classifier')
                             and values are lists of metric values over epochs.
        title (str): The title of the plot.
    """
    num_metrics = len(results_dict)
    if num_metrics == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 3), sharex=True)
    if num_metrics == 1: # If only one metric, axes is not a list
        axes = [axes]

    epochs = range(len(next(iter(results_dict.values()))))

    for i, (metric_name, values) in enumerate(results_dict.items()):
        axes[i].plot(epochs, values, label=metric_name)
        axes[i].set_ylabel(metric_name.replace('_', ' ').title())
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

def get_binary_key(outcome_tuple):
    """Converts a tuple of 0s and 1s to a binary string key."""
    return "".join(map(str, outcome_tuple))

def get_outcome_tuple(binary_key):
    """Converts a binary string key back to a tuple of ints."""
    return tuple(map(int, list(binary_key)))

def generate_all_binary_outcomes(num_vars):
    """
    Generates all possible binary outcomes for a given number of variables.
    Example: num_vars = 2 -> [(0,0), (0,1), (1,0), (1,1)]
    """
    if num_vars == 0:
        return [()]
    if num_vars == 1:
        return [(0,), (1,)]
    
    outcomes = []
    for i in range(2**num_vars):
        binary_representation = bin(i)[2:].zfill(num_vars)
        outcomes.append(tuple(map(int, list(binary_representation))))
    return outcomes


if __name__ == '__main__':
    # Test TVD
    p1_dict = {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25}
    p2_dict = {'00': 0.5, '01': 0.1, '10': 0.1, '11': 0.3}
    print(f"TVD (dict): {calculate_tvd(p1_dict, p2_dict)}") # Expected: 0.5 * (0.25 + 0.15 + 0.15 + 0.05) = 0.3

    p1_arr = np.array([0.25, 0.25, 0.25, 0.25])
    p2_arr = np.array([0.5, 0.1, 0.1, 0.3])
    print(f"TVD (array): {calculate_tvd(p1_arr, p2_arr)}") # Expected: 0.3

    # Test plotting
    dummy_results = {
        'tvd': np.random.rand(50) * 0.5,
        'born_loss': np.random.rand(50) + 0.5,
        'classifier_loss': np.random.rand(50) * 0.2
    }
    # plot_training_results(dummy_results, "Dummy Training Data") # Uncomment to test plotting

    print(f"Binary outcomes for 3 vars: {generate_all_binary_outcomes(3)}")


# classifier_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim

class BinaryClassifierMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for binary classification.
    Used in the adversarial variational inference setup to distinguish between
    samples from the Born machine q_theta(z|x)p_D(x) and the prior p(z)p_D(x).
    
    The input to the classifier is the latent variable z, and potentially the
    conditioning variable x if amortization is used.
    Input: (z, x) concatenated, or just z.
    Output: A single logit for P(sample is from class 1 (Born machine)).
    """
    def __init__(self, input_dim, hidden_dims=None, use_batch_norm=False):
        """
        Args:
            input_dim (int): Dimension of the input (e.g., num_latent_vars + conditioning_dim).
            hidden_dims (list of int, optional): List of sizes for hidden layers.
                                                 Defaults to [64, 32].
            use_batch_norm (bool): Whether to use BatchNorm1d after linear layers (before activation).
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [max(input_dim*2, 32), max(input_dim, 16)] # Default hidden layers

        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.3)) # Optional dropout
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, 1)) # Output is a single logit

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1).
                          A sigmoid will be applied outside to get probabilities.
        """
        return self.network(x)

    def get_probs(self, x):
        """Returns the probability P(sample is from class 1)"""
        logits = self.forward(x)
        return torch.sigmoid(logits)


if __name__ == '__main__':
    # Example Usage
    num_latent_vars_z = 3
    conditioning_dim_x = 2 # Dimension of x if we are conditioning
    
    # Scenario 1: Classifier input is only z
    classifier_z_only = BinaryClassifierMLP(input_dim=num_latent_vars_z, hidden_dims=[16, 8])
    print("Classifier (z only):")
    print(classifier_z_only)
    
    dummy_z_samples = torch.randn(10, num_latent_vars_z) # Batch of 10 samples
    logits_z_only = classifier_z_only(dummy_z_samples)
    probs_z_only = torch.sigmoid(logits_z_only)
    print("Logits (z only, batch_size=10):\n", logits_z_only)
    print("Probs P(class_1|z) (z only, batch_size=10):\n", probs_z_only)

    # Scenario 2: Classifier input is concatenation of (z, x)
    # This is relevant if the distributions q(z|x) and p(z) are compared *for a given x*.
    # The paper's adversarial setup (Eq. 4) has d_phi(z, x), implying x is an input.
    # The samples are (z,x) ~ q_theta(z|x)p_D(x) vs (z,x) ~ p(z)p_D(x).
    # So the classifier input should indeed include x.
    
    classifier_zx = BinaryClassifierMLP(input_dim=num_latent_vars_z + conditioning_dim_x, 
                                        hidden_dims=[32, 16])
    print("\nClassifier (z, x):")
    print(classifier_zx)

    dummy_x_condition = torch.randn(10, conditioning_dim_x) # Batch of 10 conditioning vars
    
    # Concatenate z and x for the classifier input
    classifier_input_zx = torch.cat((dummy_z_samples, dummy_x_condition), dim=1)
    print("Shape of concatenated (z,x) input:", classifier_input_zx.shape)

    logits_zx = classifier_zx(classifier_input_zx)
    probs_zx = torch.sigmoid(logits_zx)
    print("Logits (z,x):\n", logits_zx)
    print("Probs P(class_1|z,x):\n", probs_zx)

    # Test training step (conceptual)
    optimizer = optim.Adam(classifier_zx.parameters(), lr=0.001)
    # Dummy labels: 5 from class 0, 5 from class 1
    labels = torch.cat((torch.zeros(5,1), torch.ones(5,1)), dim=0) 
    
    # Loss function: Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss() # Numerically more stable than Sigmoid + BCELoss

    optimizer.zero_grad()
    output_logits = classifier_zx(classifier_input_zx) # Using the same dummy input
    loss = criterion(output_logits, labels)
    loss.backward()
    optimizer.step()
    print(f"\nExample training step: Loss = {loss.item()}")


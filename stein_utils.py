# stein_utils.py
import torch
import numpy as np
from utils import generate_all_binary_outcomes # Assuming this is in your main utils.py

def flip_bit(z_tuple, index):
    """Flips the bit at the given index in a binary tuple."""
    z_list = list(z_tuple)
    z_list[index] = 1 - z_list[index]
    return tuple(z_list)

def hamming_distance_torch(z1_tensor, z2_tensor):
    """
    Computes Hamming distance (L1 norm for binary vectors) between batches of z1 and z2.
    Args:
        z1_tensor (torch.Tensor): Shape (batch_dim, num_vars) or (num_vars,)
        z2_tensor (torch.Tensor): Shape (batch_dim, num_vars) or (num_vars,)
    Returns:
        torch.Tensor: Hamming distances, shape (batch_dim,) or scalar.
    """
    return torch.sum(torch.abs(z1_tensor - z2_tensor), dim=-1)

def base_hamming_kernel_torch(z1_tensor, z2_tensor, num_vars, length_scale=1.0):
    """
    Computes the base Hamming kernel k(z1, z2) = exp(-||z1-z2||_1 / (num_vars * length_scale)).
    The paper uses k(z,z') = exp[-(1/n)||z-z'||_1]. Here, length_scale can be 1.0 for that.

    Args:
        z1_tensor (torch.Tensor): Batch of first binary vectors. Shape (batch_size, num_vars) or (num_vars,).
        z2_tensor (torch.Tensor): Batch of second binary vectors. Shape (batch_size, num_vars) or (num_vars,).
                                  If z1 and z2 are single vectors, batch_size is effectively 1.
        num_vars (int): Number of variables (dimensionality of z).
        length_scale (float): length scale parameter for the kernel. Paper implies (1/n), so default 1.0 makes it exp(-dist/n).
                              If we want exp(-dist / (n * sigma)), then length_scale = sigma.
                              Let's stick to the paper: exp(-dist/n). So length_scale = 1.0 for the denominator.
    Returns:
        torch.Tensor: Kernel values. Shape (batch_size,) or scalar.
    """
    if num_vars == 0:
        return torch.tensor(1.0, device=z1_tensor.device, dtype=z1_tensor.dtype) # Kernel is 1 if no vars
    distance = hamming_distance_torch(z1_tensor, z2_tensor)
    return torch.exp(-distance / (num_vars * length_scale))


def compute_prob_joint_xz(bn, x_dict, z_tuple, latent_vars_names, observed_vars_names, device='cpu'):
    """
    Computes p(x, z) by marginalizing out any other variables in the Bayesian Network.
    Args:
        bn (BayesianNetwork): The Bayesian network instance.
        x_dict (dict): Observed variables, e.g., {'W': 1}.
        z_tuple (tuple): Latent variable assignment, e.g., (0, 1, 0).
        latent_vars_names (list): Ordered names of latent variables.
        observed_vars_names (list): Ordered names of observed variables.
    Returns:
        float: p(x, z)
    """
    current_assignment_dict_for_joint = dict(x_dict)
    for idx, name in enumerate(latent_vars_names):
        current_assignment_dict_for_joint[name] = z_tuple[idx]

    prob_xz = 0.0
    other_bn_vars = [node for node in bn.nodes if node not in current_assignment_dict_for_joint]
    num_other_bn_vars = len(other_bn_vars)

    if num_other_bn_vars == 0:
        # All BN vars are covered by x_obs and z_sample
        try:
            ordered_assign_tuple = [current_assignment_dict_for_joint[name] for name in bn.nodes]
            prob_xz = bn.get_joint_probability(tuple(ordered_assign_tuple))
        except KeyError as e:
            raise KeyError(f"Node {e} not found in current_assignment_dict_for_joint. BN nodes: {bn.nodes}, assignment: {current_assignment_dict_for_joint}")

    else:
        # Marginalize out other_bn_vars
        all_other_assignments = generate_all_binary_outcomes(num_other_bn_vars)
        for other_assignment_tuple in all_other_assignments:
            temp_full_assign_dict = dict(current_assignment_dict_for_joint)
            for i_other, other_name in enumerate(other_bn_vars):
                temp_full_assign_dict[other_name] = other_assignment_tuple[i_other]
            
            ordered_assign_tuple = [temp_full_assign_dict[name] for name in bn.nodes]
            prob_xz += bn.get_joint_probability(tuple(ordered_assign_tuple))
    return prob_xz


def get_score_function_sp_for_z(bn, x_dict, z_tuple, latent_vars_names, observed_vars_names, device='cpu'):
    """
    Computes the score function s_p(x, z) for a single z_tuple.
    s_p(x,z)_i = 1 - p(x, neg_i z) / p(x,z)

    Args:
        bn (BayesianNetwork): The Bayesian Network.
        x_dict (dict): Observed variables.
        z_tuple (tuple): The specific assignment of latent variables.
        latent_vars_names (list): Ordered names of latent variables.
        observed_vars_names (list): Ordered names of observed variables.
        device (str): PyTorch device.

    Returns:
        torch.Tensor: Score vector s_p(x,z) of shape (num_latent_vars,).
    """
    num_latent_vars = len(latent_vars_names)
    s_p_vector = torch.zeros(num_latent_vars, device=device, dtype=torch.float32)

    # p(x,z)
    prob_x_z = compute_prob_joint_xz(bn, x_dict, z_tuple, latent_vars_names, observed_vars_names, device=device)

    if prob_x_z < 1e-9: # If p(x,z) is effectively zero
        # Score is ill-defined or can be set to a default (e.g., 0 or 1).
        # The paper assumes p(z|x) > 0, which implies p(x,z) > 0 if p(x) > 0.
        # If p(x,z) is zero, using this z in KSD might be problematic.
        # For now, let's return a vector that won't cause division by zero, e.g. zeros.
        # print(f"Warning: p(x,z) is near zero for x={x_dict}, z={z_tuple}. Score function might be unstable.")
        return s_p_vector # Or handle as per specific KSD implementation requirements

    for i in range(num_latent_vars):
        z_neg_i = flip_bit(z_tuple, i)
        prob_x_z_neg_i = compute_prob_joint_xz(bn, x_dict, z_neg_i, latent_vars_names, observed_vars_names, device=device)
        
        s_p_vector[i] = 1.0 - (prob_x_z_neg_i / prob_x_z)
        
    return s_p_vector

def get_stein_kernel_kp_value(
    z1_tuple, z2_tuple, x_dict, 
    bn, latent_vars_names, observed_vars_names,
    base_kernel_func, # e.g., partial(base_hamming_kernel_torch, num_vars=N, length_scale=1.0)
    sp_at_z1, sp_at_z2, # Precomputed score functions s_p(x,z1) and s_p(x,z2)
    device='cpu'):
    """
    Computes a single value of the Stein kernel k_p(z1, z2 | x) (Eq. 13).

    Args:
        z1_tuple (tuple): First latent variable assignment.
        z2_tuple (tuple): Second latent variable assignment.
        x_dict (dict): Observed variables.
        bn (BayesianNetwork): The Bayesian network.
        latent_vars_names (list): Ordered names of latent variables.
        observed_vars_names (list): Ordered names of observed variables.
        base_kernel_func (callable): Function base_kernel_func(tensor_z1, tensor_z2) -> scalar tensor.
        sp_at_z1 (torch.Tensor): Score s_p(x,z1), shape (num_latent_vars,).
        sp_at_z2 (torch.Tensor): Score s_p(x,z2), shape (num_latent_vars,).
        device (str): PyTorch device.

    Returns:
        torch.Tensor: Scalar value of k_p(z1, z2 | x).
    """
    num_latent_vars = len(latent_vars_names)
    if num_latent_vars == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32) # Or 1.0 if it's more like a product

    z1 = torch.tensor(z1_tuple, dtype=torch.float32, device=device)
    z2 = torch.tensor(z2_tuple, dtype=torch.float32, device=device)

    # Term 1: s_p(x,z1)^T k(z1,z2) s_p(x,z2)
    k_z1_z2 = base_kernel_func(z1, z2)
    term1 = torch.dot(sp_at_z1, sp_at_z2) * k_z1_z2

    # Term 2: -s_p(x,z1)^T Delta_z2 k(z1,z2)
    # Delta_z2_k is a vector: j-th element is k(z1,z2) - k(z1, neg_j z2)
    delta_z2_k_vec = torch.zeros(num_latent_vars, device=device, dtype=torch.float32)
    for j in range(num_latent_vars):
        z2_neg_j = torch.tensor(flip_bit(z2_tuple, j), dtype=torch.float32, device=device)
        delta_z2_k_vec[j] = k_z1_z2 - base_kernel_func(z1, z2_neg_j)
    term2 = -torch.dot(sp_at_z1, delta_z2_k_vec)

    # Term 3: - (Delta_z1 k(z1,z2))^T s_p(x,z2)
    # Delta_z1_k is a vector: i-th element is k(z1,z2) - k(neg_i z1, z2)
    delta_z1_k_vec = torch.zeros(num_latent_vars, device=device, dtype=torch.float32)
    for i in range(num_latent_vars):
        z1_neg_i = torch.tensor(flip_bit(z1_tuple, i), dtype=torch.float32, device=device)
        delta_z1_k_vec[i] = k_z1_z2 - base_kernel_func(z1_neg_i, z2)
    term3 = -torch.dot(delta_z1_k_vec, sp_at_z2)
    
    # Term 4: + tr[Delta_z1,z2 k(z1,z2)]
    # Delta_z1_z2_k is a matrix: (i,j)-th element is
    # k(z1,z2) - k(z1, neg_j z2) - k(neg_i z1, z2) + k(neg_i z1, neg_j z2)
    trace_term4 = torch.tensor(0.0, device=device, dtype=torch.float32)
    if num_latent_vars > 0 : # Trace is only for i=j
        for i in range(num_latent_vars): # Diagonal elements for trace
            z1_neg_i = torch.tensor(flip_bit(z1_tuple, i), dtype=torch.float32, device=device)
            z2_neg_i = torch.tensor(flip_bit(z2_tuple, i), dtype=torch.float32, device=device) # For tr, j=i
            
            k_z1_z2_neg_i = base_kernel_func(z1, z2_neg_i)
            k_z1_neg_i_z2 = base_kernel_func(z1_neg_i, z2)
            k_z1_neg_i_z2_neg_i = base_kernel_func(z1_neg_i, z2_neg_i)
            
            diag_element = k_z1_z2 - k_z1_z2_neg_i - k_z1_neg_i_z2 + k_z1_neg_i_z2_neg_i
            trace_term4 += diag_element
    
    kp_value = term1 + term2 + term3 + trace_term4
    return kp_value

if __name__ == '__main__':
    # Basic tests
    print("Testing stein_utils...")
    device = 'cpu'
    
    # flip_bit
    assert flip_bit((0,0,0), 0) == (1,0,0)
    assert flip_bit((1,0,1), 1) == (1,1,1)
    print("flip_bit OK")

    # hamming_distance_torch
    z1 = torch.tensor([0,0,1,1], dtype=torch.float32)
    z2 = torch.tensor([1,0,0,1], dtype=torch.float32)
    dist = hamming_distance_torch(z1,z2)
    assert dist.item() == 2.0
    z1_b = torch.tensor([[0,0],[1,1]], dtype=torch.float32)
    z2_b = torch.tensor([[0,1],[1,0]], dtype=torch.float32)
    dist_b = hamming_distance_torch(z1_b, z2_b)
    assert torch.allclose(dist_b, torch.tensor([1.,1.]))
    print("hamming_distance_torch OK")

    # base_hamming_kernel_torch
    N_VARS = 4
    val = base_hamming_kernel_torch(z1, z2, N_VARS, length_scale=1.0)
    expected_val = np.exp(-2.0/N_VARS)
    assert torch.isclose(val, torch.tensor(expected_val, dtype=torch.float32))
    print("base_hamming_kernel_torch OK")

    # Dummy BN for testing score and kp (A -> B)
    from bayesian_network import BayesianNetwork # Assuming it's in the same directory or path
    test_bn = BayesianNetwork()
    test_bn.add_node('A', cpt={(): {0:0.8, 1:0.2}}) # P(A=1)=0.2
    test_bn.add_node('B', cpt={(0,):{0:0.7, 1:0.3}, (1,):{0:0.4, 1:0.6}}, parent_names=['A'])
    # Latent: A, Observed: B
    
    lat_vars = ['A']
    obs_vars = ['B']
    x_obs = {'B':1}
    z_test = (1,) # A=1
    
    # Test compute_prob_joint_xz
    # p(B=1, A=1) = P(B=1|A=1)P(A=1) = 0.6 * 0.2 = 0.12
    # p(B=1, A=0) = P(B=1|A=0)P(A=0) = 0.3 * 0.8 = 0.24
    p_b1_a1 = compute_prob_joint_xz(test_bn, x_obs, (1,), lat_vars, obs_vars)
    assert np.isclose(p_b1_a1, 0.12)
    p_b1_a0 = compute_prob_joint_xz(test_bn, x_obs, (0,), lat_vars, obs_vars)
    assert np.isclose(p_b1_a0, 0.24)
    print("compute_prob_joint_xz OK")

    # Test get_score_function_sp_for_z
    # s_p(B=1, A=1)_A = 1 - p(B=1, A=0)/p(B=1,A=1) = 1 - 0.24/0.12 = 1 - 2 = -1
    sp_A1_B1 = get_score_function_sp_for_z(test_bn, x_obs, (1,), lat_vars, obs_vars, device=device)
    assert torch.isclose(sp_A1_B1[0], torch.tensor(-1.0))
    # s_p(B=1, A=0)_A = 1 - p(B=1, A=1)/p(B=1,A=0) = 1 - 0.12/0.24 = 1 - 0.5 = 0.5
    sp_A0_B1 = get_score_function_sp_for_z(test_bn, x_obs, (0,), lat_vars, obs_vars, device=device)
    assert torch.isclose(sp_A0_B1[0], torch.tensor(0.5))
    print("get_score_function_sp_for_z OK")

    # Test get_stein_kernel_kp_value (simple case, 1 latent var)
    from functools import partial
    base_kernel_1d = partial(base_hamming_kernel_torch, num_vars=1, length_scale=1.0)
    
    z1_t = (0,) # A=0
    z2_t = (1,) # A=1
    
    sp_z1 = sp_A0_B1 # s_p(x, A=0)
    sp_z2 = sp_A1_B1 # s_p(x, A=1)

    # k_p(A=0, A=1 | B=1)
    kp_01 = get_stein_kernel_kp_value(z1_t, z2_t, x_obs, test_bn, lat_vars, obs_vars, base_kernel_1d, sp_z1, sp_z2, device=device)
    # For 1D:
    # k_01 = exp(-1/1) = exp(-1)
    # s0 = 0.5, s1 = -1
    # t1 = s0*s1*k_01 = 0.5 * -1 * exp(-1) = -0.5 * exp(-1)
    # Delta_z2 k(0,1) = k(0,1) - k(0,0) = exp(-1) - exp(0) = exp(-1) - 1
    # t2 = -s0 * (exp(-1)-1) = -0.5 * (exp(-1)-1)
    # Delta_z1 k(0,1) = k(0,1) - k(1,1) = exp(-1) - exp(0) = exp(-1) - 1
    # t3 = -(exp(-1)-1) * s1 = -(exp(-1)-1) * -1 = exp(-1)-1
    # Delta_z1z2 k(0,1) = k(0,1) - k(0,0) - k(1,1) + k(1,0)
    #                   = exp(-1) - 1 - 1 + exp(-1) = 2*exp(-1) - 2
    # t4 = 2*exp(-1) - 2
    # kp = -0.5e-1 -0.5e-1 + 0.5 + e-1 -1 + 2e-1 -2 = (-0.5-0.5+1+2)e-1 + (0.5-1-2) = 2e-1 - 2.5
    expected_kp_01 = 2*np.exp(-1) - 2.5 
    assert torch.isclose(kp_01, torch.tensor(expected_kp_01, dtype=torch.float32)), f"kp_01: {kp_01.item()}, expected: {expected_kp_01}"

    # k_p(A=0, A=0 | B=1)
    kp_00 = get_stein_kernel_kp_value(z1_t, z1_t, x_obs, test_bn, lat_vars, obs_vars, base_kernel_1d, sp_z1, sp_z1, device=device)
    # k_00 = exp(0) = 1
    # t1 = s0*s0*k_00 = 0.5*0.5*1 = 0.25
    # Delta_z2 k(0,0) = k(0,0) - k(0,1) = 1 - exp(-1)
    # t2 = -s0 * (1-exp(-1)) = -0.5 * (1-exp(-1))
    # Delta_z1 k(0,0) = k(0,0) - k(1,0) = 1 - exp(-1)
    # t3 = -(1-exp(-1))*s0 = -0.5 * (1-exp(-1))
    # Delta_z1z2 k(0,0) = k(0,0) - k(0,1) - k(1,0) + k(1,1)
    #                   = 1 - exp(-1) - exp(-1) + 1 = 2 - 2*exp(-1)
    # t4 = 2 - 2*exp(-1)
    # kp = 0.25 -0.5(1-e-1) -0.5(1-e-1) + 2 - 2e-1
    #    = 0.25 -0.5 +0.5e-1 -0.5 +0.5e-1 + 2 - 2e-1
    #    = (0.25 - 1 + 2) + (0.5+0.5-2)e-1 = 1.25 - e-1
    expected_kp_00 = 1.25 - np.exp(-1)
    assert torch.isclose(kp_00, torch.tensor(expected_kp_00, dtype=torch.float32)), f"kp_00: {kp_00.item()}, expected: {expected_kp_00}"
    print("get_stein_kernel_kp_value OK (for 1D example)")
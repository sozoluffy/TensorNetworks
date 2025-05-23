# stein_utils.py
import torch
import numpy as np
# Assuming generate_all_binary_outcomes is in your main utils.py
from utils import generate_all_binary_outcomes 

def flip_bit(z_tuple, index):
    """Flips the bit at the given index in a binary tuple."""
    z_list = list(z_tuple)
    z_list[index] = 1 - z_list[index]
    return tuple(z_list)

def hamming_distance_torch(z1_tensor, z2_tensor):
    """
    Computes Hamming distance (L1 norm for binary vectors) between batches of z1 and z2.
    Input tensors should have the same dtype.
    """
    # Ensure both tensors are on the same device before operation
    if z1_tensor.device != z2_tensor.device:
        z2_tensor = z2_tensor.to(z1_tensor.device)
    # Ensure both tensors have the same dtype
    if z1_tensor.dtype != z2_tensor.dtype:
        # Promote to the higher precision dtype or default to float64
        common_dtype = torch.float64 if torch.float64 in (z1_tensor.dtype, z2_tensor.dtype) else torch.float32
        z1_tensor = z1_tensor.to(common_dtype)
        z2_tensor = z2_tensor.to(common_dtype)
        
    return torch.sum(torch.abs(z1_tensor - z2_tensor), dim=-1)

def base_hamming_kernel_torch(z1_tensor, z2_tensor, num_vars, length_scale=1.0):
    """
    Computes the base Hamming kernel k(z1, z2) = exp(-||z1-z2||_1 / (num_vars * length_scale)).
    The paper uses k(z,z') = exp[-(1/n)||z-z'||_1]. Here, length_scale can be 1.0 for that.
    Output dtype matches input dtype (promoted if necessary).
    """
    if num_vars == 0:
        # Determine dtype from inputs or default to float64
        dtype_to_use = z1_tensor.dtype if hasattr(z1_tensor, 'dtype') else (z2_tensor.dtype if hasattr(z2_tensor, 'dtype') else torch.float64)
        device_to_use = z1_tensor.device if hasattr(z1_tensor, 'device') else (z2_tensor.device if hasattr(z2_tensor, 'device') else 'cpu')
        return torch.tensor(1.0, device=device_to_use, dtype=dtype_to_use)

    # Ensure z1_tensor and z2_tensor have compatible dtypes and devices for hamming_distance_torch
    common_dtype = torch.float64 if torch.float64 in (z1_tensor.dtype, z2_tensor.dtype) else torch.float32
    if z1_tensor.device != z2_tensor.device:
        z2_tensor = z2_tensor.to(z1_tensor.device)
    z1_casted = z1_tensor.to(common_dtype)
    z2_casted = z2_tensor.to(common_dtype)
    
    distance = hamming_distance_torch(z1_casted, z2_casted)
    # Ensure denominator is float and has the same dtype for division
    denominator = torch.tensor(float(num_vars) * float(length_scale), device=distance.device, dtype=distance.dtype)
    if denominator == 0: # Avoid division by zero if num_vars or length_scale is 0
        # If distance is also 0, kernel is 1. If distance > 0, kernel is 0.
        return torch.ones_like(distance) if torch.all(distance == 0) else torch.zeros_like(distance)
    return torch.exp(-distance / denominator)


def compute_prob_joint_xz(bn, x_dict, z_tuple, latent_vars_names, observed_vars_names, device='cpu'):
    """
    Computes p(x, z) by marginalizing out any other variables in the Bayesian Network.
    """
    current_assignment_dict_for_joint = dict(x_dict) if x_dict else {} 
    for idx, name in enumerate(latent_vars_names):
        current_assignment_dict_for_joint[name] = z_tuple[idx]

    prob_xz = 0.0
    other_bn_vars = [node for node in bn.nodes if node not in current_assignment_dict_for_joint]
    num_other_bn_vars = len(other_bn_vars)

    if num_other_bn_vars == 0:
        final_assignment_values = []
        all_keys_present = True
        for node_name_in_order in bn.nodes:
            if node_name_in_order not in current_assignment_dict_for_joint:
                all_keys_present = False
                break 
            final_assignment_values.append(current_assignment_dict_for_joint[node_name_in_order])
        
        if all_keys_present and len(final_assignment_values) == len(bn.nodes):
             prob_xz = bn.get_joint_probability(tuple(final_assignment_values))
        else: 
            # This path indicates an issue, num_other_bn_vars should not be 0 if not all keys covered
            # Force marginalization logic by ensuring num_other_bn_vars is non-zero effectively
            # This indicates a mismatch between bn.nodes and the union of latent/observed/other.
            # For safety, we recalculate other_bn_vars based on ALL bn.nodes vs current_assignment
            # However, the initial calculation of other_bn_vars should be correct.
            # This branch implies an unexpected state; robust code would re-evaluate other_bn_vars.
            # For now, if this happens, it's likely a setup error.
            # We re-set num_other_bn_vars to a non-zero if all_keys_present is false
            # to ensure the marginalization loop is entered if something is inconsistent.
            if not all_keys_present:
                 num_other_bn_vars = -1 # Indicate error/force marginalization path for safety
                 other_bn_vars = [node for node in bn.nodes if node not in current_assignment_dict_for_joint] # Recalculate

    if num_other_bn_vars != 0: 
        all_other_assignments = generate_all_binary_outcomes(len(other_bn_vars))
        for other_assignment_tuple in all_other_assignments:
            temp_full_assign_dict = dict(current_assignment_dict_for_joint)
            for i_other, other_name in enumerate(other_bn_vars):
                temp_full_assign_dict[other_name] = other_assignment_tuple[i_other]
            
            ordered_assign_tuple_values = []
            valid_assignment = True
            for node_name_in_order in bn.nodes:
                if node_name_in_order not in temp_full_assign_dict:
                    valid_assignment = False
                    break
                ordered_assign_tuple_values.append(temp_full_assign_dict[node_name_in_order])

            if valid_assignment:
                prob_xz += bn.get_joint_probability(tuple(ordered_assign_tuple_values))
    return float(prob_xz) # Return Python float for broader compatibility


def get_score_function_sp_for_z(bn, x_dict, z_tuple, latent_vars_names, observed_vars_names, device='cpu'):
    """
    Computes the score function s_p(x, z) for a single z_tuple.
    s_p(x,z)_i = 1 - p(x, neg_i z) / p(x,z)
    Returns torch.float64 tensor.
    """
    num_latent_vars = len(latent_vars_names)
    s_p_vector = torch.zeros(num_latent_vars, device=device, dtype=torch.float64)

    prob_x_z = compute_prob_joint_xz(bn, x_dict, z_tuple, latent_vars_names, observed_vars_names, device=device)

    if abs(prob_x_z) < 1e-12: 
        # print(f"Warning: p(x,z) is near zero ({prob_x_z}) for x={x_dict}, z={z_tuple}. Score vector set to zeros.")
        return s_p_vector 

    for i in range(num_latent_vars):
        z_neg_i = flip_bit(z_tuple, i)
        prob_x_z_neg_i = compute_prob_joint_xz(bn, x_dict, z_neg_i, latent_vars_names, observed_vars_names, device=device)
        
        s_p_vector[i] = 1.0 - (prob_x_z_neg_i / prob_x_z) # Python floats will be cast to float64
        
    return s_p_vector

def get_stein_kernel_kp_value(
    z1_tuple, z2_tuple, x_dict, 
    bn, latent_vars_names, observed_vars_names,
    base_kernel_func, # e.g., partial(base_hamming_kernel_torch, num_vars=N, length_scale=1.0)
    sp_at_z1, sp_at_z2, # Expected to be torch.float64 from ksd_vi_quantum.py
    device='cpu'):
    """
    Computes a single value of the Stein kernel k_p(z1, z2 | x) (Eq. 13).
    Ensures all internal tensors and calculations use torch.float64.
    """
    num_latent_vars = len(latent_vars_names)
    target_dtype = torch.float64 # Explicitly use float64 for all calculations

    if num_latent_vars == 0:
        return torch.tensor(0.0, device=device, dtype=target_dtype)

    z1 = torch.tensor(z1_tuple, dtype=target_dtype, device=device)
    z2 = torch.tensor(z2_tuple, dtype=target_dtype, device=device)

    # Ensure score functions are target_dtype (should be float64 from ksd_vi_quantum)
    sp_at_z1 = sp_at_z1.to(dtype=target_dtype, device=device)
    sp_at_z2 = sp_at_z2.to(dtype=target_dtype, device=device)

    # Term 1: s_p(x,z1)^T k(z1,z2) s_p(x,z2)
    # base_kernel_func should ideally also return target_dtype
    k_z1_z2 = base_kernel_func(z1, z2).to(target_dtype) 
    term1 = torch.dot(sp_at_z1, sp_at_z2) * k_z1_z2

    # Term 2: -s_p(x,z1)^T Delta_z2 k(z1,z2)
    delta_z2_k_vec = torch.zeros(num_latent_vars, device=device, dtype=target_dtype)
    for j in range(num_latent_vars):
        z2_neg_j = torch.tensor(flip_bit(z2_tuple, j), dtype=target_dtype, device=device)
        delta_z2_k_vec[j] = k_z1_z2 - base_kernel_func(z1, z2_neg_j).to(target_dtype)
    term2 = -torch.dot(sp_at_z1, delta_z2_k_vec)

    # Term 3: - (Delta_z1 k(z1,z2))^T s_p(x,z2)
    delta_z1_k_vec = torch.zeros(num_latent_vars, device=device, dtype=target_dtype)
    for i in range(num_latent_vars):
        z1_neg_i = torch.tensor(flip_bit(z1_tuple, i), dtype=target_dtype, device=device)
        delta_z1_k_vec[i] = k_z1_z2 - base_kernel_func(z1_neg_i, z2).to(target_dtype)
    term3 = -torch.dot(delta_z1_k_vec, sp_at_z2)
    
    # Term 4: + tr[Delta_z1,z2 k(z1,z2)]
    trace_term4 = torch.tensor(0.0, device=device, dtype=target_dtype)
    if num_latent_vars > 0 :
        for i in range(num_latent_vars): 
            z1_neg_i = torch.tensor(flip_bit(z1_tuple, i), dtype=target_dtype, device=device)
            # For trace, second index of Delta is also i for the diagonal element
            z2_neg_i_for_diag = torch.tensor(flip_bit(z2_tuple, i), dtype=target_dtype, device=device) 
            
            k_z1_z2_neg_i_diag = base_kernel_func(z1, z2_neg_i_for_diag).to(target_dtype)
            k_z1_neg_i_z2 = base_kernel_func(z1_neg_i, z2).to(target_dtype) # This z2 is original z2
            k_z1_neg_i_z2_neg_i_diag = base_kernel_func(z1_neg_i, z2_neg_i_for_diag).to(target_dtype)
            
            # Delta_z1_i,z2_i k(z1,z2) = k(z1,z2) - k(z1, neg_i z2) - k(neg_i z1, z2) + k(neg_i z1, neg_i z2)
            diag_element = k_z1_z2 - k_z1_z2_neg_i_diag - k_z1_neg_i_z2 + k_z1_neg_i_z2_neg_i_diag
            trace_term4 += diag_element
    
    kp_value = term1 + term2 + term3 + trace_term4
    return kp_value

if __name__ == '__main__':
    # (Keep the tests from the previous version, ensuring they use torch.float64 where appropriate)
    print("Testing stein_utils with explicit torch.float64 dtypes...")
    device = 'cpu'
    target_dtype = torch.float64 

    assert flip_bit((0,0,0), 0) == (1,0,0)
    print("flip_bit OK")

    z1_tst = torch.tensor([0,0,1,1], dtype=target_dtype, device=device)
    z2_tst = torch.tensor([1,0,0,1], dtype=target_dtype, device=device)
    dist = hamming_distance_torch(z1_tst,z2_tst)
    assert dist.item() == 2.0, f"Hamming distance failed: {dist.item()}"
    print("hamming_distance_torch OK")

    N_VARS_TST = 4
    val_kernel = base_hamming_kernel_torch(z1_tst, z2_tst, N_VARS_TST, length_scale=1.0)
    expected_val_kernel = np.exp(-2.0/N_VARS_TST) # np.exp returns float64
    assert torch.isclose(val_kernel.to(target_dtype), torch.tensor(expected_val_kernel, dtype=target_dtype)), f"Kernel val: {val_kernel}, expected: {expected_val_kernel}"
    print("base_hamming_kernel_torch OK")

    from bayesian_network import BayesianNetwork 
    test_bn = BayesianNetwork()
    test_bn.add_node('A', cpt={(): {0:0.8, 1:0.2}}) 
    test_bn.add_node('B', cpt={(0,):{0:0.7, 1:0.3}, (1,):{0:0.4, 1:0.6}}, parent_names=['A'])
    
    lat_vars_tst = ['A']
    obs_vars_tst = ['B']
    x_obs_tst = {'B':1}
    
    p_b1_a1_tst = compute_prob_joint_xz(test_bn, x_obs_tst, (1,), lat_vars_tst, obs_vars_tst)
    assert np.isclose(p_b1_a1_tst, 0.12), f"p_b1_a1_tst: {p_b1_a1_tst}"
    print("compute_prob_joint_xz OK")

    sp_A1_B1_tst = get_score_function_sp_for_z(test_bn, x_obs_tst, (1,), lat_vars_tst, obs_vars_tst, device=device)
    assert torch.isclose(sp_A1_B1_tst[0], torch.tensor(-1.0, dtype=target_dtype)), f"sp_A1_B1_tst: {sp_A1_B1_tst[0]}"
    sp_A0_B1_tst = get_score_function_sp_for_z(test_bn, x_obs_tst, (0,), lat_vars_tst, obs_vars_tst, device=device)
    assert torch.isclose(sp_A0_B1_tst[0], torch.tensor(0.5, dtype=target_dtype)), f"sp_A0_B1_tst: {sp_A0_B1_tst[0]}"
    print("get_score_function_sp_for_z OK")

    from functools import partial
    base_kernel_1d_tst = partial(base_hamming_kernel_torch, num_vars=1, length_scale=1.0)
    
    z1_tuple_tst = (0,) 
    z2_tuple_tst = (1,) 
    
    kp_01_tst = get_stein_kernel_kp_value(z1_tuple_tst, z2_tuple_tst, x_obs_tst, test_bn, lat_vars_tst, obs_vars_tst, base_kernel_1d_tst, sp_A0_B1_tst, sp_A1_B1_tst, device=device)
    expected_kp_01_tst = 2*np.exp(-1.0) - 2.5 
    assert torch.isclose(kp_01_tst, torch.tensor(expected_kp_01_tst, dtype=target_dtype)), f"kp_01: {kp_01_tst.item()}, expected: {expected_kp_01_tst}"

    kp_00_tst = get_stein_kernel_kp_value(z1_tuple_tst, z1_tuple_tst, x_obs_tst, test_bn, lat_vars_tst, obs_vars_tst, base_kernel_1d_tst, sp_A0_B1_tst, sp_A0_B1_tst, device=device)
    expected_kp_00_tst = 1.25 - np.exp(-1.0)
    assert torch.isclose(kp_00_tst, torch.tensor(expected_kp_00_tst, dtype=target_dtype)), f"kp_00: {kp_00_tst.item()}, expected: {expected_kp_00_tst}"
    print("get_stein_kernel_kp_value OK (for 1D example with float64)")
    print("All stein_utils tests passed with corrected dtypes!")
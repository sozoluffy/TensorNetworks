# bayesian_network.py
import numpy as np
from collections import defaultdict
from utils import get_binary_key, get_outcome_tuple, generate_all_binary_outcomes

class BayesianNetwork:
    """
    Represents a simple Bayesian Network with binary variables.
    Nodes are identified by string names.
    CPTs (Conditional Probability Tables) define P(Node | Parents).
    """
    def __init__(self):
        self.nodes = []  # List of node names, defines an ordering
        self.parents = defaultdict(list) # node_name -> [parent_name_1, parent_name_2, ...]
        self.cpts = {} # node_name -> CPT_object or callable
        self.node_to_index = {}

    def add_node(self, name, cpt, parent_names=None):
        """
        Adds a node to the Bayesian Network.

        Args:
            name (str): The name of the node.
            cpt (dict or callable): The Conditional Probability Table for this node.
                If dict:
                    Keys are tuples representing parent states (e.g., (0,1) for two parents).
                    Values are dicts P(Node=0|Parents) and P(Node=1|Parents).
                    Example for node 'C' with parent 'A':
                    { (0,): {0: 0.8, 1: 0.2}, # P(C=0|A=0)=0.8, P(C=1|A=0)=0.2
                      (1,): {0: 0.3, 1: 0.7}  # P(C=0|A=1)=0.3, P(C=1|A=1)=0.7 }
                    For nodes with no parents, key is an empty tuple ():
                    { (): {0: 0.9, 1: 0.1} } # P(C=0)=0.9
                If callable:
                    A function that takes a tuple of parent values and returns
                    a dict {0: prob_node_is_0, 1: prob_node_is_1}.
            parent_names (list of str, optional): List of names of parent nodes.
                                                 Order matters for CPT keys.
        """
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists.")

        self.nodes.append(name)
        self.node_to_index[name] = len(self.nodes) - 1

        if parent_names:
            for parent_name in parent_names:
                if parent_name not in self.nodes:
                    raise ValueError(f"Parent node {parent_name} for {name} not found. Add parents first.")
            self.parents[name] = list(parent_names) # Ensure it's a list

        self.cpts[name] = cpt

    def _get_prob_node_given_parents(self, node_name, parent_values_tuple):
        """
        Gets P(node_name=1 | parent_values_tuple).
        """
        cpt_entry = self.cpts[node_name]
        if callable(cpt_entry):
            prob_dict = cpt_entry(parent_values_tuple)
        elif isinstance(cpt_entry, dict):
            prob_dict = cpt_entry.get(parent_values_tuple)
            if prob_dict is None:
                raise ValueError(f"CPT entry for node {node_name} with parent values {parent_values_tuple} not found.")
        else:
            raise TypeError(f"CPT for node {node_name} has an invalid type.")

        if not isinstance(prob_dict, dict) or 0 not in prob_dict or 1 not in prob_dict:
             raise ValueError(f"CPT for {node_name} with parent values {parent_values_tuple} must return a dict {{0: p0, 1: p1}}")
        
        # Ensure probabilities sum to 1 (approximately)
        if not np.isclose(prob_dict[0] + prob_dict[1], 1.0):
            raise ValueError(f"Probabilities for node {node_name} given parents {parent_values_tuple} do not sum to 1: {prob_dict}")
            
        return prob_dict[1] # P(Node=1)

    def sample_forward(self, num_samples=1):
        """
        Performs forward sampling from the Bayesian Network.
        Assumes nodes are added in a topological order (parents before children).

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            list of dicts: Each dict represents a sample, e.g., {'A':0, 'B':1, 'C':0}
                           Order of keys in dict might not be guaranteed.
            list of tuples: Each tuple represents a sample in the order of self.nodes.
        """
        samples_dict_list = []
        samples_tuple_list = []

        for _ in range(num_samples):
            current_sample_dict = {}
            current_sample_tuple = [0] * len(self.nodes)

            for node_name in self.nodes: # Assumes topological order
                parent_names = self.parents[node_name]
                parent_values = tuple(current_sample_dict[p_name] for p_name in parent_names)

                prob_node_is_1 = self._get_prob_node_given_parents(node_name, parent_values)
                
                sampled_value = 1 if np.random.rand() < prob_node_is_1 else 0
                current_sample_dict[node_name] = sampled_value
                current_sample_tuple[self.node_to_index[node_name]] = sampled_value
            
            samples_dict_list.append(current_sample_dict)
            samples_tuple_list.append(tuple(current_sample_tuple))
            
        return samples_dict_list, samples_tuple_list

    def get_joint_probability(self, full_assignment_tuple):
        """
        Calculates the joint probability of a full assignment of variables.
        P(X1=x1, X2=x2, ..., Xn=xn)

        Args:
            full_assignment_tuple (tuple): A tuple of values (0 or 1) for all variables
                                           in the order of self.nodes.

        Returns:
            float: The joint probability.
        """
        if len(full_assignment_tuple) != len(self.nodes):
            raise ValueError("Full assignment tuple length must match the number of nodes.")

        # Create a dictionary from the tuple for easier lookup
        assignment_dict = {node_name: full_assignment_tuple[self.node_to_index[node_name]] for node_name in self.nodes}
        
        joint_prob = 1.0
        for node_name in self.nodes: # Assumes topological order
            node_value = assignment_dict[node_name]
            parent_names = self.parents[node_name]
            parent_values_tuple = tuple(assignment_dict[p_name] for p_name in parent_names)

            cpt_entry = self.cpts[node_name]
            if callable(cpt_entry):
                prob_dict = cpt_entry(parent_values_tuple)
            else: # dict
                prob_dict = cpt_entry.get(parent_values_tuple)
            
            if prob_dict is None:
                raise ValueError(f"CPT entry for node {node_name} with parent values {parent_values_tuple} not found.")

            joint_prob *= prob_dict[node_value] # P(Node=node_value | Parents=parent_values)
            
        return joint_prob

    def get_true_posterior(self, latent_vars_names, observed_vars_dict):
        """
        Calculates the true posterior distribution P(Latent | Observed) by enumeration.
        This is computationally expensive for larger networks.

        Args:
            latent_vars_names (list of str): Names of the latent variables.
            observed_vars_dict (dict): Dictionary of observed_var_name -> value.

        Returns:
            dict: Posterior distribution. Keys are tuples of latent variable assignments (in order of latent_vars_names),
                  values are probabilities.
            float: Normalization constant P(Observed).
        """
        if not all(name in self.nodes for name in latent_vars_names):
            raise ValueError("One or more latent variable names not in the network.")
        if not all(name in self.nodes for name in observed_vars_dict.keys()):
            raise ValueError("One or more observed variable names not in the network.")
        
        # Check for overlap
        if set(latent_vars_names) & set(observed_vars_dict.keys()):
            raise ValueError("Latent and observed variables must be disjoint.")

        num_latent_vars = len(latent_vars_names)
        posterior_unnormalized = {}
        
        # Generate all possible assignments for latent variables
        latent_assignments = generate_all_binary_outcomes(num_latent_vars)

        for latent_assignment_tuple in latent_assignments:
            current_full_assignment_dict = dict(observed_vars_dict)
            for i, var_name in enumerate(latent_vars_names):
                current_full_assignment_dict[var_name] = latent_assignment_tuple[i]
            
            # Convert dict to tuple in the order of self.nodes for get_joint_probability
            full_assignment_tuple_ordered = [0] * len(self.nodes)
            for node_name, val in current_full_assignment_dict.items():
                full_assignment_tuple_ordered[self.node_to_index[node_name]] = val
            
            # Ensure all variables are assigned
            if len(current_full_assignment_dict) != len(self.nodes):
                missing_vars = set(self.nodes) - set(current_full_assignment_dict.keys())
                # This case should ideally not happen if latent + observed cover all,
                # or if the remaining are marginalized out. For simplicity, we assume
                # latent + observed covers all relevant variables for the joint.
                # If not, the model implies marginalization over unmentioned variables.
                # For this function, we expect latent + observed to define the slice we care about.
                # The paper's examples usually condition on some variables and infer others.
                # We will calculate P(Latent, Observed)
                
                # Reconstruct the full assignment by iterating through all possibilities of missing vars
                # This part becomes complex if there are many missing vars.
                # For now, assume latent_vars + observed_vars are the *only* variables we are interested in,
                # and the BN defines their joint. Or, that other variables are to be marginalized out.
                # The simplest interpretation for P(Latent | Observed) = P(Latent, Observed) / P(Observed)
                # is to calculate P(Latent, Observed) by fixing Latent and Observed, and marginalizing
                # any other variables.
                # However, the paper's examples (like Sprinkler) use all variables.
                # If latent + observed don't cover all nodes, this function needs to be smarter
                # or the BN definition needs to be for a sub-problem.
                # For now, let's assume latent_vars_names + observed_vars_dict.keys() == self.nodes
                all_vars_in_query = set(latent_vars_names) | set(observed_vars_dict.keys())
                if all_vars_in_query != set(self.nodes):
                    # This means there are other variables in the network not part of the query.
                    # We need to marginalize them out.
                    # This requires iterating over all combinations of these "other" variables.
                    other_vars = list(set(self.nodes) - all_vars_in_query)
                    num_other_vars = len(other_vars)
                    
                    sum_joint_prob_for_this_latent_assignment = 0.0
                    other_assignments = generate_all_binary_outcomes(num_other_vars)
                    
                    for other_assignment_tuple in other_assignments:
                        temp_full_assignment_dict = dict(current_full_assignment_dict) # has latent and observed
                        for i_other, other_var_name in enumerate(other_vars):
                            temp_full_assignment_dict[other_var_name] = other_assignment_tuple[i_other]
                        
                        # Order it
                        ordered_temp_full_assignment_tuple = [0] * len(self.nodes)
                        for node_name_temp, val_temp in temp_full_assignment_dict.items():
                            ordered_temp_full_assignment_tuple[self.node_to_index[node_name_temp]] = val_temp
                        
                        sum_joint_prob_for_this_latent_assignment += self.get_joint_probability(tuple(ordered_temp_full_assignment_tuple))
                    
                    joint_prob_latent_observed = sum_joint_prob_for_this_latent_assignment

                else: # All variables are covered by latent + observed
                     joint_prob_latent_observed = self.get_joint_probability(tuple(full_assignment_tuple_ordered))

            else: # All variables are covered by latent + observed
                joint_prob_latent_observed = self.get_joint_probability(tuple(full_assignment_tuple_ordered))

            posterior_unnormalized[latent_assignment_tuple] = joint_prob_latent_observed
            
        # Normalize
        prob_observed = sum(posterior_unnormalized.values())
        
        if prob_observed == 0:
            # This can happen if the evidence is impossible.
            # Return uniform or zeros, or raise error. For now, return zeros.
            print(f"Warning: P(Observed) is zero for evidence {observed_vars_dict}. Posterior is ill-defined.")
            posterior_normalized = {k: 0.0 for k in posterior_unnormalized.keys()}
        else:
            posterior_normalized = {k: v / prob_observed for k, v in posterior_unnormalized.items()}
            
        return posterior_normalized, prob_observed

    def get_prior_distribution(self, var_names_ordered):
        """
        Calculates the prior distribution P(var_names_ordered) by marginalizing out other variables.
        This is for the adversarial method where we need P(z).

        Args:
            var_names_ordered (list of str): Names of the variables for which to get the prior.
                                             The output keys will be tuples in this order.
        Returns:
            dict: Prior distribution. Keys are tuples of variable assignments, values are probabilities.
        """
        num_target_vars = len(var_names_ordered)
        prior_dist = defaultdict(float)

        other_vars = [node for node in self.nodes if node not in var_names_ordered]
        num_other_vars = len(other_vars)

        # Iterate over all possible assignments for the target variables
        target_assignments = generate_all_binary_outcomes(num_target_vars)

        for target_assignment_tuple in target_assignments:
            current_assignment_dict = {var_names_ordered[i]: target_assignment_tuple[i] for i in range(num_target_vars)}
            
            sum_prob_for_this_target_assignment = 0.0

            if num_other_vars > 0:
                # Iterate over all possible assignments for the 'other' variables to marginalize them out
                other_assignments = generate_all_binary_outcomes(num_other_vars)
                for other_assignment_tuple in other_assignments:
                    full_assignment_dict = dict(current_assignment_dict)
                    for i, var_name in enumerate(other_vars):
                        full_assignment_dict[var_name] = other_assignment_tuple[i]
                    
                    # Order the full assignment according to self.nodes
                    full_assignment_tuple_ordered = [0] * len(self.nodes)
                    for node_name, val in full_assignment_dict.items():
                        full_assignment_tuple_ordered[self.node_to_index[node_name]] = val
                    
                    sum_prob_for_this_target_assignment += self.get_joint_probability(tuple(full_assignment_tuple_ordered))
            else: # No other variables, target_vars are all variables
                 full_assignment_tuple_ordered = [0] * len(self.nodes)
                 for node_name, val in current_assignment_dict.items():
                    full_assignment_tuple_ordered[self.node_to_index[node_name]] = val
                 sum_prob_for_this_target_assignment += self.get_joint_probability(tuple(full_assignment_tuple_ordered))

            prior_dist[target_assignment_tuple] = sum_prob_for_this_target_assignment
        
        # Sanity check: probabilities should sum to 1
        if not np.isclose(sum(prior_dist.values()), 1.0):
            print(f"Warning: Prior probabilities for {var_names_ordered} sum to {sum(prior_dist.values())}, not 1.0.")

        return dict(prior_dist)


# --- Example: Sprinkler Network from the paper ---
# P(C,S,R,W) = P(C)P(S|C)P(R|C)P(W|S,R)
# Variables: C (Cloudy), S (Sprinkler), R (Rain), W (Grass Wet)
def get_sprinkler_network(random_cpts=False):
    """
    Creates the Sprinkler Bayesian Network.
    Nodes: Cloudy (C), Sprinkler (S), Rain (R), Grass Wet (W)
    Structure: C -> S, C -> R, (S,R) -> W
    Order for this implementation: C, S, R, W
    """
    bn = BayesianNetwork()

    if random_cpts:
        # For random CPTs similar to paper's Fig 6 setup
        def random_p(): return np.random.uniform(0.01, 0.99)
        
        # Cloudy (C)
        p_c_true = random_p()
        bn.add_node('C', cpt={(): {0: 1-p_c_true, 1: p_c_true}})
        
        # Sprinkler (S) | Cloudy (C)
        p_s_given_c_false = random_p()
        p_s_given_c_true = random_p()
        bn.add_node('S', cpt={
            (0,): {0: 1-p_s_given_c_false, 1: p_s_given_c_false}, # C=0
            (1,): {0: 1-p_s_given_c_true,  1: p_s_given_c_true}   # C=1
        }, parent_names=['C'])

        # Rain (R) | Cloudy (C)
        p_r_given_c_false = random_p()
        p_r_given_c_true = random_p()
        bn.add_node('R', cpt={
            (0,): {0: 1-p_r_given_c_false, 1: p_r_given_c_false}, # C=0
            (1,): {0: 1-p_r_given_c_true,  1: p_r_given_c_true}   # C=1
        }, parent_names=['C'])

        # Grass Wet (W) | Sprinkler (S), Rain (R)
        # Order of parents for CPT: (S, R)
        p_w_sf_rf = random_p() # S=0, R=0
        p_w_sf_rt = random_p() # S=0, R=1
        p_w_st_rf = random_p() # S=1, R=0
        p_w_st_rt = random_p() # S=1, R=1
        bn.add_node('W', cpt={
            (0,0): {0: 1-p_w_sf_rf, 1: p_w_sf_rf}, # S=0, R=0
            (0,1): {0: 1-p_w_sf_rt, 1: p_w_sf_rt}, # S=0, R=1
            (1,0): {0: 1-p_w_st_rf, 1: p_w_st_rf}, # S=1, R=0
            (1,1): {0: 1-p_w_st_rt, 1: p_w_st_rt}  # S=1, R=1
        }, parent_names=['S', 'R'])

    else: # Standard textbook probabilities
        # Cloudy (C)
        bn.add_node('C', cpt={(): {0: 0.5, 1: 0.5}})
        
        # Sprinkler (S) | Cloudy (C)
        bn.add_node('S', cpt={
            (0,): {0: 0.5, 1: 0.5}, # C=0 (Not Cloudy) -> P(S=1)=0.5 (e.g. scheduled)
            (1,): {0: 0.9, 1: 0.1}  # C=1 (Cloudy)    -> P(S=1)=0.1 (less likely to turn on)
        }, parent_names=['C'])

        # Rain (R) | Cloudy (C)
        bn.add_node('R', cpt={
            (0,): {0: 0.8, 1: 0.2}, # C=0 (Not Cloudy) -> P(R=1)=0.2
            (1,): {0: 0.2, 1: 0.8}  # C=1 (Cloudy)    -> P(R=1)=0.8
        }, parent_names=['C'])

        # Grass Wet (W) | Sprinkler (S), Rain (R)
        # Order of parents for CPT: (S, R)
        bn.add_node('W', cpt={
            (0,0): {0: 0.99, 1: 0.01}, # S=0, R=0 -> P(W=1)=0.01 (dry)
            (0,1): {0: 0.1,  1: 0.9},  # S=0, R=1 -> P(W=1)=0.9  (wet due to rain)
            (1,0): {0: 0.1,  1: 0.9},  # S=1, R=0 -> P(W=1)=0.9  (wet due to sprinkler)
            (1,1): {0: 0.01, 1: 0.99}  # S=1, R=1 -> P(W=1)=0.99 (very wet)
        }, parent_names=['S', 'R'])
        
    return bn

if __name__ == '__main__':
    sprinkler_bn = get_sprinkler_network(random_cpts=False)
    print("Nodes:", sprinkler_bn.nodes)
    print("Node to index:", sprinkler_bn.node_to_index)
    print("Parents of W:", sprinkler_bn.parents['W'])
    print("CPT for C:", sprinkler_bn.cpts['C'])
    print("CPT for W[(0,0)] (S=0,R=0):", sprinkler_bn.cpts['W'][(0,0)])

    # Test sampling
    print("\n--- Sampling ---")
    _, samples_tuples = sprinkler_bn.sample_forward(5)
    for i, s_tuple in enumerate(samples_tuples):
        s_dict = {sprinkler_bn.nodes[j]: s_tuple[j] for j in range(len(s_tuple))}
        print(f"Sample {i+1} (dict): {s_dict}")
        print(f"Sample {i+1} (tuple): {s_tuple}, Joint P: {sprinkler_bn.get_joint_probability(s_tuple):.4f}")

    # Test posterior calculation: P(C,S,R | W=1)
    # Latent vars: C, S, R. Observed: W=1
    print("\n--- True Posterior P(C,S,R | W=1) ---")
    latent_vars = ['C', 'S', 'R'] # Order for posterior keys
    observed_vars = {'W': 1}
    
    true_posterior, p_observed = sprinkler_bn.get_true_posterior(latent_vars, observed_vars)
    
    print(f"P(Observed = {observed_vars}) = {p_observed:.4f}")
    print("Posterior P(C,S,R | W=1):")
    for assignment_tuple, prob in sorted(true_posterior.items()):
        assignment_dict = {latent_vars[i]: assignment_tuple[i] for i in range(len(assignment_tuple))}
        print(f"  P({assignment_dict} | {observed_vars}) = {prob:.4f}")
    
    if not np.isclose(sum(true_posterior.values()), 1.0) and p_observed > 0:
        print(f"Warning: Posterior probabilities sum to {sum(true_posterior.values())}")

    # Test prior calculation P(C,S,R)
    print("\n--- Prior P(C,S,R) ---")
    prior_vars_csr = ['C', 'S', 'R']
    prior_dist_csr = sprinkler_bn.get_prior_distribution(prior_vars_csr)
    print(f"Prior P({','.join(prior_vars_csr)}):")
    for assignment_tuple, prob in sorted(prior_dist_csr.items()):
         assignment_dict = {prior_vars_csr[i]: assignment_tuple[i] for i in range(len(assignment_tuple))}
         print(f"  P({assignment_dict}) = {prob:.4f}")
    if not np.isclose(sum(prior_dist_csr.values()), 1.0):
        print(f"Warning: Prior CSR probabilities sum to {sum(prior_dist_csr.values())}")


    # Test with a different network (simpler)
    # A -> B
    print("\n--- Simple BN A->B ---")
    simple_bn = BayesianNetwork()
    simple_bn.add_node('A', cpt={(): {0: 0.7, 1: 0.3}})
    simple_bn.add_node('B', cpt={(0,): {0:0.9, 1:0.1}, (1,): {0:0.2, 1:0.8}}, parent_names=['A'])

    # P(A | B=1)
    lat_simple = ['A']
    obs_simple = {'B': 1}
    post_simple, p_obs_simple = simple_bn.get_true_posterior(lat_simple, obs_simple)
    print(f"P(B=1) = {p_obs_simple:.4f}")
    for assignment_tuple, prob in sorted(post_simple.items()):
        print(f"  P(A={assignment_tuple[0]} | B=1) = {prob:.4f}")

    # P(A)
    prior_A = simple_bn.get_prior_distribution(['A'])
    print(f"Prior P(A): {prior_A}")


    # Test random CPTs for Sprinkler
    print("\n--- Sprinkler with Random CPTs ---")
    random_sprinkler_bn = get_sprinkler_network(random_cpts=True)
    _, r_samples_tuples = random_sprinkler_bn.sample_forward(1)
    r_s_tuple = r_samples_tuples[0]
    r_s_dict = {random_sprinkler_bn.nodes[j]: r_s_tuple[j] for j in range(len(r_s_tuple))}
    print(f"Sample (dict): {r_s_dict}")
    print(f"Sample (tuple): {r_s_tuple}, Joint P: {random_sprinkler_bn.get_joint_probability(r_s_tuple):.4f}")

    r_latent_vars = ['C', 'S', 'R']
    r_observed_vars = {'W': 1}
    r_true_posterior, r_p_observed = random_sprinkler_bn.get_true_posterior(r_latent_vars, r_observed_vars)
    print(f"P(Observed = {r_observed_vars}) for random CPTs = {r_p_observed:.4f}")
    if r_p_observed > 0 and not np.isclose(sum(r_true_posterior.values()), 1.0) :
         print(f"Warning: Random CPT Posterior probabilities sum to {sum(r_true_posterior.values())}")



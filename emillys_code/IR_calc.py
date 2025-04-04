from math import log2

def compute_information_density(joint_probs, marginal_probs):
    """
    Computes Information Density (ID) using joint and marginal probabilities.
    Formula: ID = - Σ p(x,y) log₂ (p(x,y) / p(x))
    """
    ID = 0.0
    for prefix, suffix_probs in joint_probs.items():
        for suffix, p_xy in suffix_probs.items():
            p_x = marginal_probs[prefix]  # p(x)
            ID += -p_xy * log2(p_xy / p_x)  

    return ID
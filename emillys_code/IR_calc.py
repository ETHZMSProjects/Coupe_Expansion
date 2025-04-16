from math import log2
import pandas as pd
import warnings

def compute_information_density(joint_probs, marginal_probs):
    """
    Computes Information Density (ID) using joint and marginal probabilities.
    Formula: ID = - Σ p(x,y) log₂ (p(x,y) / p(x))

    Args:
        joint_probs (dict): Joint probabilities from model
        marginal_probs (dict): Marginal probabilities from model

    Returns:
        float: Normalized ID value
    """
    baseline = vietnamese_ID_for_normalization()
    ID = 0.0
    for prefix, suffix_probs in joint_probs.items():
        p_x = marginal_probs[prefix]  # p(x)
        if p_x == 0:
            continue  # skip to avoid division by zero

        for suffix, p_xy in suffix_probs.items():
            if p_xy > 0:
                ID -= p_xy * log2(p_xy / p_x)

    if baseline:
        return ID / baseline
    else:
        warnings.warn("Warning: Baseline for ID normalization is not provided. Returning ID = 0.0.")
        return 0.0 


def vietnamese_ID_for_normalization(): 
    """
    Loads a tab-separated CSV file and extracts all Information Density (ID) values 
    for Vietnamese speakers. Prints each ID and computes the average ID, which can 
    be used as a baseline for normalization.

    The function assumes:
    - The CSV file is tab-separated (`\t`)
    - There's a column named 'Language' with 'VIE' representing Vietnamese
    - There's a column named 'ID' containing the information density values

    Returns:
        float: The average ID value for Vietnamese speakers
    """
    df = pd.read_csv(r"C:\Users\emill\Documents\GitHub\Coupe_Expansion\InfoRateData.csv", sep='\t')

    # Filter rows where the Language is Vietnamese
    vietnamese_df = df[df['Language'] == 'VIE'] 

    #print("All Vietnamese ID values:")
    #print(vietnamese_df['ID'])

    # Compute the average ID for Vietnamese speakers
    ID_vietnamese = vietnamese_df['ID'].mean()

    print("Baseline Vietnamese ID:", ID_vietnamese)

    return ID_vietnamese
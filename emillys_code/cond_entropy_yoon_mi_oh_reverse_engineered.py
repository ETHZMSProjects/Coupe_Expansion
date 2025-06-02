import math
from collections import defaultdict
import pandas as pd
import numpy as np
import re
from helpers import clean_ipa

    
def log2(n):
    return math.log(n) / math.log(2)


def compute_cond_entropy(file_path, lang_cfg):
    """
    Computes:
      - Unigram entropy H(X)
      - Bigram conditional entropy H(Y|X)
    Returns:
      tuple: (ID_unigram, ID_bigram)
    """

    columns = lang_cfg["Columns"] 

    # Initialize variables
    hash_map = defaultdict(int)
    total = 0

    # Read and process bigram frequency data
    if lang_cfg.name in ["FRA", "DEU"]:
        # Load appropriate file format
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            df = pd.read_csv(file_path, sep="\t", encoding="utf-8")

        # Filter rows if a column filter is defined
        if pd.notna(columns[1]) and columns[1] in df.columns:
            df = df[df[columns[1]] > 0]

        for _, row in df.iterrows():
            raw_word = str(row[columns[0]])
            freq = float(row[columns[1]])
            hash_map[raw_word] += freq
            total += freq

    else:  # Text-based format (e.g. ENG, CMN, VIE, JPN, YUE)
        with open(file_path, 'r') as file:
            for line in file:
                word, freq = line.strip().split('\t')
                freq = float(freq)
                if freq > 0:
                    hash_map[word] += freq
                    total += freq


    # Unigram entropy (ID_unigram) (added by esidaine)
    probs_unigram = {w: c / total for w, c in hash_map.items()}
    ID_unigram = -sum(p * log2(p) for p in probs_unigram.values())

    # Calculate bigram frequencies
    count = defaultdict(int)
    syll_delimiter = lang_cfg["Syllable Delimiter"]
    if isinstance(syll_delimiter, str):
        if syll_delimiter.startswith("[") and syll_delimiter.endswith("]"):
            syll_delimiter = syll_delimiter[1:-1]
        
    for raw_word in sorted(hash_map):
        syllables = re.split(f"[{syll_delimiter}]", raw_word)

        # Clean each syllable individually (cleaning added by esidaine)
        cleaned_syllables = [
            "".join(clean_ipa(s, False, 'sylls', syll_delimiter, lang_cfg.name))
            for s in syllables if s.strip()
        ]
        cleaned_syllables = [s for s in cleaned_syllables if s] # Remove empty strings
    
        for i in range(len(cleaned_syllables) - 1):
            bigram = (cleaned_syllables[i], cleaned_syllables[i + 1])
            count[bigram] += hash_map[raw_word]
    
    # Compute prefix counts
    prefix_counts = defaultdict(int)
    for (x, y), freq in count.items():
        prefix_counts[x] += freq

    # Normalize bigram frequencies
    normalized = defaultdict(float)
    for (x, y), freq in count.items():
        if prefix_counts[x] > 0.0:
            normalized[(x, y)] = freq / prefix_counts[x]

    #  Compute conditional entropy
    entropy_map = defaultdict(float)
    for bigram, prob in normalized.items():
        if prob > 0.0:
            entropy_map[bigram] = -prob * log2(prob) # Apply the Shannon entropy formula to each bigram

    #  Sum entropy for each starting syllable
    sum_map = defaultdict(float)
    for bigram, entropy in entropy_map.items():
        x,y = bigram
        sum_map[x] += entropy


    # Compute weighted average of conditional entropy for the language 
    total_prefix = sum(prefix_counts.values())
    prob_prefix = {x: prefix_counts[x] / total_prefix for x in prefix_counts}
    ID_bigram = sum(prob_prefix[x] * sum_map.get(x, 0) for x in prob_prefix)

    print("Total bigrams:", len(count))
    print("Most frequent bigram:", max(count, key=count.get))
    print(f"Total syllable tokens: {total}")
    print(f"Unique sequences: {len(hash_map)}")
    print(f"Unique bigrams: {len(count)}")
    print(f"ID_unigram: {ID_unigram:.3f}, ID_bigram: {ID_bigram:.3f}")
        
    return ID_unigram, ID_bigram

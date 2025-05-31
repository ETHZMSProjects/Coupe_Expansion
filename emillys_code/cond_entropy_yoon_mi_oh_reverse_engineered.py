import math
from collections import defaultdict

    
def log2(n):
    return math.log(n) / math.log(2)


def compute_cond_entropy(file_path):
    """
    Computes:
      - Unigram entropy H(X)
      - Bigram conditional entropy H(Y|X)
    Returns:
      tuple: (ID_unigram, ID_bigram)
    """

    # Initialize variables
    hash_map = defaultdict(int)
    total = 0

    # Read and process bigram frequency data
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

    # Calculate stats
    type_count = len(hash_map)
    most = max(hash_map, key=hash_map.get)
    most_freq = hash_map[most]
    hapax = sum(1 for v in hash_map.values() if v == 1)

    # Calculate bigram frequencies
    count = defaultdict(int)
    for word in sorted(hash_map):
        syllables = word.split('_')
        for i in range (len(syllables) - 1):
            bigram = (syllables[i], syllables[i + 1])
            count[bigram] += hash_map[word]
    

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
    print(f"{file_path} — Total syllable tokens: {total}")
    print(f"Unique sequences: {len(hash_map)}")
    print(f"Unique bigrams: {len(count)}")
    print(f"ID_unigram: {ID_unigram:.3f}, ID_bigram: {ID_bigram:.3f}")
        
    return ID_unigram, ID_bigram

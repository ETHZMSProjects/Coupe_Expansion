import math
from collections import defaultdict

def log2(n):
    return math.log(n) / math.log(2)

input_file = "vie.txt"
output_file = "conditionalEntropyBigrams.txt"

# Initialize variables
hash_map = defaultdict(int)
total = 0

# Read and process bigram frequency data
with open(input_file, 'r') as file:
    for line in file:
        word, freq = line.strip().split('\t')
        freq = int(freq)
        if freq > 0:
            hash_map[word] += freq
            total += freq

# Calculate stats
type_count = len(hash_map)
most = max(hash_map, key=hash_map.get)
most_freq = hash_map[most]
hapax = sum(1 for v in hash_map.values() if v == 1)

with open(output_file, 'w') as out:
    print(f"Total frequency of bigram (token) is {total}\n")
    print(f"Total number of bigram (type) is {type_count} \n")
    print(f"Most frequent bigram is {most} with {most_freq} ocurrences\n")
    #print(f"The number of hapax is {hapax}\n")

    # Step 1: Calculate bigram frequencies
    count = defaultdict(int)
    for word in sorted(hash_map):
        syllables = word.split('_')
        for i in range (len(syllables) - 1):
            bigram = (syllables[i], syllables[i + 1])
            count[bigram] += hash_map[word]
    
    print(count.items()) 

    # Compute prefix counts
    prefix_counts = defaultdict(int)
    for (x, y), freq in count.items():
        prefix_counts[x] += freq

    # Step 2: Normalize bigram frequencies
    normalized = defaultdict(float)

    for (x, y), freq in count.items():
        if prefix_counts[x] > 0.0:
            normalized[(x, y)] = freq / prefix_counts[x]
    print("Normalized bigram frequencies:")
    for (x, y), prob in normalized.items():
        print(f"P({y}|{x}) = {prob:.4f}")

    #  Step 3: Compute conditional entropy
    entropy_map = defaultdict(float)
    for bigram, prob in normalized.items():
        if prob > 0.0:
            entropy_map[bigram] = -prob * log2(prob) # Apply the Shannon entropy formula to each bigram
            # print(f"Bigram: {bigram}, Entropy: {entropy_map[bigram]}")

    #  Step 4: Sum entropy for each starting syllable
    sum_map = defaultdict(float)
    for bigram, entropy in entropy_map.items():
        x,y = bigram
        sum_map[x] += entropy
        # print(f"Prefix: {x}, Sum Entropy: {sum_map[x]}")

    # Step 5: Compute total entropy for the language by summing the weighted entropies of each prefix

    prob_prefix  = {x: prefix_counts[x] / total for x in prefix_counts}
    # print(f"Probability of each bigram: {prob_prefix}")

    total_entropy = 0.0
    for x in sorted(prob_prefix):
        total_entropy += prob_prefix[x] * sum_map.get(x, 0)

    print(f"The value of conditional entropy for {input_file} is {total_entropy}\n")

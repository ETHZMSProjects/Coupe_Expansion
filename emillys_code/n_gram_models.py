import nltk
from nltk.util import ngrams
from collections import defaultdict
from math import log2

def generate_ngrams(transcribed_sentence, n):
    """
    Generate n-grams from a list of syllables.
    """
    return list(ngrams(transcribed_sentence, n, pad_left=True, pad_right=True, left_pad_symbol="<BOS>", right_pad_symbol="<EOS>"))


def build_markov_chain(merged_sentences, n):
    """
    Build an n-gram transition probability model.
    """
    ngram_counts = defaultdict(lambda: defaultdict(int)) # maps prefix to next token e.g. {('I', 'am'): {'happy': 3, 'tired': 2}}
    total_ngrams = 0 

    # Create n-grams
    ngram_list = generate_ngrams(merged_sentences, n)

    # Count occurrences of each n-gram
    for gram in ngram_list:
        prefix, next_token = tuple(gram[:-1]), gram[-1]
        ngram_counts[prefix][next_token] += 1
        total_ngrams += 1
    
    # Convert counts to probabilities
    joint_probs  = defaultdict(dict)
    marginal_probs = defaultdict(float)

    for prefix, suffix_counts in ngram_counts.items():
        prefix_occurences = sum(suffix_counts.values()) # total ocurrences of the prefix across all possible suffixes
        marginal_probs[prefix] = prefix_occurences / total_ngrams # p(x)
        
        for suffix, suffix_count in suffix_counts.items():
            joint_probs [prefix][suffix] = suffix_count / total_ngrams   # p(x, y) 


    return joint_probs, marginal_probs
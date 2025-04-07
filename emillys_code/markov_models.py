import nltk
from nltk.util import ngrams
from collections import defaultdict
from math import log2
import os
import pickle


class MarkovModel: 
    def __init__(self, n): 
        """
        Initialize an n-gram Markov model.
        """
        self.n = n # n-gram size
        self.joint_probs = defaultdict(dict)  # Stores p(x, y) — how often two items appear together
        self.marginal_probs = defaultdict(float) # Stores p(x) — how often the first part shows up

    def build(self, sentences):
        """
        Build the n-gram model from a list of transcribed sentences
        """
        ngram_counts = defaultdict(lambda: defaultdict(int))
        total_ngrams = 0

        # Create n-grams
        ngram_list = self.generate_ngrams(sentences)

         # Count occurrences of each n-gram
        for gram in ngram_list:
            prefix, next_token = tuple(gram[:-1]), gram[-1]
            ngram_counts[prefix][next_token] += 1
            total_ngrams += 1

        for prefix, suffix_counts in ngram_counts.items():
            prefix_total = sum(suffix_counts.values()) # total ocurrences of the prefix across all possible suffixes
            self.marginal_probs[prefix] = prefix_total / total_ngrams # p(x)
            for suffix, count in suffix_counts.items():
                self.joint_probs[prefix][suffix] = count / total_ngrams # p(x, y) 
    
    def generate_ngrams(self, sentence_list):
        """
        Generate padded n-grams from a list of syllables/words.
        """
        return list(ngrams(sentence_list, self.n, pad_left=True, pad_right=True, 
                           left_pad_symbol="<BOS>", right_pad_symbol="<EOS>"))

    def save(self, language):
        """
        Save this specific model
        """
        os.makedirs(f"produced_data/{language}", exist_ok=True)

        with open(f"produced_data/{language}/{language}_markov_model_{self.n}gram.pkl", "wb") as f:
            pickle.dump(self, f)

        print(f"\n✅ Saved {self.n}-gram model to 'produced_data/{language}/'")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


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
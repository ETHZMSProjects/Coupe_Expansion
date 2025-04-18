import nltk
from nltk.util import ngrams
from collections import defaultdict
from math import log2
import os
import pickle
import warnings


class MarkovModel: 
    def __init__(self, n): 
        """
        Initialize an n-gram Markov model.
        """
        self.n = n # n-gram size
        self.ngram_counts = defaultdict(int)  # Bigram or n-gram counts
        self.prefix_counts = defaultdict(int)  # Prefix counts for normalization
        self.normalized_probs = defaultdict(float)
        self.entropy_map = defaultdict(float) # entropy_map[prefix] holds H(Y|X=prefix)
        self.total_ngrams = 0

    def build(self, input, input_type):
        """
        Build the n-gram model from syllable-level sentence data or within-word syllables.
        `input_type` must be either 'sentences' or 'words'.
        """
        ngram_list = []

        if input_type == "sentences":  
            # Create n-grams
            ngram_list = self.generate_ngrams_for_sent(input)

        elif input_type == "words":
            ngram_list = self.generate_ngrams_for_words(input)   
        else: 
            warnings.warn("Unknown input_type. Use 'sentences' or 'words'.")
            return
        
        # Count occurrence of each n-gram and prefix
        for gram in ngram_list:
            prefix, next_token = tuple(gram[:-1]), gram[-1]

            self.ngram_counts[(prefix, next_token)] += 1
            self.prefix_counts[prefix] += 1
            self.total_ngrams += 1

        # Normalize to conditional probabilities P(y|x) = P(x,y) / P(x)
        for (prefix, next_token), count in self.ngram_counts.items():
            self.normalized_probs[(prefix, next_token)] = count / self.prefix_counts[prefix]

        # Compute entropy for each prefix
        for (prefix, next_token), prob in self.normalized_probs.items():
            if prob > 0:
                # accumulate –prob·log₂(prob) for each next_token
                self.entropy_map[prefix] += - prob * log2(prob)


        """
        print(f"normalized_probs: {self.normalized_probs}")
        print(f"entropy_map: {self.entropy_map}")
        print(f"prefix_counts: {self.prefix_counts}")
        print(f"ngram_counts: {self.ngram_counts}")
        """
    

    def compute_conditional_entropy(self):
        """
        Compute the conditional entropy for each n-gram in the model.
        """
        total_entropy = 0.0
        for prefix, prefix_entropy in self.entropy_map.items():
            p_prefix = self.prefix_counts[prefix] / self.total_ngrams  # = P(x)
            total_entropy += p_prefix * prefix_entropy # sum of P(x) * H(y|x) for all prefixes
        return total_entropy

    def generate_ngrams_for_sent(self, input_list):
        """
        Generate padded n-grams from a list of syllables.
        input_list is a flat list of all sentences in the data
        """
        return list(ngrams(input_list, self.n))

    def generate_ngrams_for_words(self, input_list):
        """
        Generate n-grams within each word only (no cross-word transitions).
        input_list is a list of syllable lists (i.e., list of words).

        Example: 
        input_list = [['tə5', 'lə5'], ['wə3'], ['i1', 'Qai4']] 
        """
        ngrams_per_word = []
        for syllables_per_word in input_list: 
            if len(syllables_per_word) >= self.n: 
                word_ngrams  = list(ngrams(syllables_per_word, self.n))
                ngrams_per_word.extend(word_ngrams)
        return ngrams_per_word

    def save_model(self, language, input_type):
        """
        Save this specific model
        """
        os.makedirs(f"produced_data/{language}", exist_ok=True)

        with open(f"produced_data/{language}/{language}_{input_type}_markov_model_{self.n}gram.pkl", "wb") as f:
            pickle.dump(self, f)

        print(f"\n✅ Saved {self.n}-gram model to 'produced_data/{language}/'")



    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
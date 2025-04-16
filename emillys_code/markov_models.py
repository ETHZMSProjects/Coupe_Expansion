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
        self.joint_probs = defaultdict(dict)  # Stores p(x, y) — how often two items appear together
        self.marginal_probs = defaultdict(float) # Stores p(x) — how often the first part shows up

    def build(self, input, input_type):
        """
        Build the n-gram model from syllable-level sentence data or within-word syllables.
        `input_type` must be either 'sentences' or 'words'.
        """
        ngram_counts = defaultdict(lambda: defaultdict(int))
        total_ngrams = 0

        if input_type == "sentences":  
            # Create n-grams
            ngram_list = self.generate_ngrams_for_sent(input)

        elif input_type == "words":
            ngram_list = self.generate_ngrams_for_words(input)
        
        else: 
            warnings.warn("Warning: input_type is unknown, MarkovModel cannot be built.")
            return

        print(ngram_list)
        
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


    
    def generate_ngrams_for_sent(self, input_list):
        """
        Generate padded n-grams from a list of syllables.
        input_list is a flat list of all sentences in the data
        """
        return list(ngrams(input_list, self.n, pad_left=True, pad_right=True, 
                           left_pad_symbol="<BOS>", right_pad_symbol="<EOS>"))

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

    def save(self, language, input_type):
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
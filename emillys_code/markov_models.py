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
        self.cond_probs = defaultdict(float)
        self.entropy_map = defaultdict(float) # entropy_map[prefix] holds H(Y|X=prefix)
        self.total_ngrams = 0
        self.total_prefixes = 0

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

        self.total_prefixes = len(self.prefix_counts)

        # print("----joint probs-----")
        for (prefix, next_token), joint_count in self.ngram_counts.items():
            marginal_prob = self.prefix_counts[prefix] / self.total_ngrams
            joint_prob = joint_count / self.total_ngrams  # P(x,y)
            self.cond_probs[(prefix, next_token)] = joint_prob / marginal_prob # P(y|x)
    

    def compute_conditional_entropy(self):
        """
        Compute the conditional entropy H(Y|X) = Σ P(x) * H(Y|X=x)
        where:
            - P(x) = prefix frequency / total ngrams
            - P(y|x) = count(x, y) / count(x)
            - H(Y|X=x) = - Σ P(y|x) * log2(P(y|x))
        """
        prefix_entropy = defaultdict(float)

        # Compute H(Y|X)
        for (prefix, next_token), joint_count in self.ngram_counts.items():
                cond_prob = self.cond_probs[(prefix, next_token)]  # P(y|x)
                if cond_prob > 0: # Omit non-ocurring pairs to avoid log2(0)
                    prefix_entropy[prefix] += cond_prob * (- log2(cond_prob)) # H(Y|X=x) = -Σ P(y|x) * log2(P(y|x))


        # Compute the overall entropy by weighting H(Y|X=x) by P(x)
        total_entropy = 0.0
        for prefix, cond_entropy in prefix_entropy.items():
            marginal_prob = self.prefix_counts[prefix] / self.total_ngrams  # P(x)
            total_entropy += marginal_prob * cond_entropy # H(Y|X) = Σ P(x) * H(Y|X=x)

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
        print(f"ngram examples: {ngrams_per_word[:5]}")
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
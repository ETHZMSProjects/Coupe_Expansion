from nltk.util import ngrams
from collections import defaultdict
from math import log2
import os
import pickle
import warnings
import logging

logging.basicConfig(level=logging.INFO)


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
        self.total_prefixes = 0 # Total number of unique prefixes

    def build(self, data, text_type):
        """
        Build ngrams for within word transitions or sentence-level transitions at the level of the chosen linguistic unit.
        Applies additive smoothing (Dirichlet smoothing).

        text_type must be either 'sentences' or 'words'.
        """
        ngram_list = []
        alpha = 0.1

        if text_type == "sentences":  
            ngram_list = self.generate_ngrams_for_sent(data)
        elif text_type == "words":
            ngram_list = self.generate_ngrams_for_words(data)
        else: 
            warnings.warn("Unknown text_type. Use 'sentences' or 'words'.")
            return
        
        # Count occurrence of each n-gram and prefix
        for gram in ngram_list:
            prefix, next_token = tuple(gram[:-1]), gram[-1]

            self.ngram_counts[(prefix, next_token)] += 1
            self.prefix_counts[prefix] += 1
            self.total_ngrams += 1

        self.total_prefixes = len(self.prefix_counts)

        # Build vocabulary of possible next tokens
        vocabulary = set(y for (_, y) in self.ngram_counts.keys())
        vocab_size = len(vocabulary)

        # Conditional probabilities with additive smoothing
        for prefix in self.prefix_counts:
            prefix_count = self.prefix_counts[prefix]
            for next_token in vocabulary: 
                count_xy = self.ngram_counts.get((prefix, next_token), 0)
                smoothed_prob = (count_xy + alpha) / (prefix_count + alpha * vocab_size)
                self.cond_probs[(prefix, next_token)] = smoothed_prob
    

    def compute_conditional_entropy(self):
        """
        Compute the conditional entropy H(Y|X) = Σ P(x) * H(Y|X=x)
        where:
            - P(x) = prefix frequency / total ngrams
            - P(y|x) = count(x, y) / count(x)
            - H(Y|X=x) = - Σ P(y|x) * log2(P(y|x))
        """
        prefix_entropy = defaultdict(float)

        # Compute H(Y|X) for each prefix
        for (prefix, next_token), cond_prob in self.cond_probs.items():
                if cond_prob > 0: # Avoid log2(0)
                    prefix_entropy[prefix] += cond_prob * (-log2(cond_prob)) # H(Y|X=x) = -Σ P(y|x) * log2(P(y|x))


        # Compute overall entropy by weighting H(Y|X=x) by P(x)
        total_entropy = 0.0
        total_prefix_count = sum(self.prefix_counts.values())  # total count of prefixes

        for prefix, cond_entropy in prefix_entropy.items():
            marginal_prob = self.prefix_counts[prefix] / total_prefix_count  # P(x)
            total_entropy += marginal_prob * cond_entropy # H(Y|X) = Σ P(x) * H(Y|X=x)

        return total_entropy

    def generate_ngrams_for_sent(self, input_list):
        """
        Generate n-grams across words within each sentence, but not across sentences.
        
        input_list is a List of sentences:
        - each sentence is List of words
        - each word is List of phonemes, segments or syllables
        
        Example:
        input_list = [
            [ ['p','u','ʁ'], ['l','ə'], ['ʁ','w','a','j','o','m'] ],  # sentence 1
            [ ['l','e'], ['ɡ','ɛ','ʁ'], ['d','ə'] ],                  # sentence 2
            ...
        ]
        
        The function concatenates the words within each sentence:
        sentence_stream = ['p','u','ʁ','l','ə','ʁ','w','a','j','o','m']
        Then makes n-grams over this stream.
        """
        ngrams_per_sentence = []

        for sentence in input_list:
            # Flatten sentence to stream of linguistic units
            sentence_stream = [segment for word in sentence for segment in word]
            
            # Generate n-grams if enough units
            if len(sentence_stream) >= self.n:
                sentence_ngrams = list(ngrams(sentence_stream, self.n))
                ngrams_per_sentence.extend(sentence_ngrams)

        logging.info(f"ngram examples: {ngrams_per_sentence[:5]}")
        return ngrams_per_sentence

    def generate_ngrams_for_words(self, input_list):
        """
        Generate n-grams within each word only (no cross-word transitions).
        Supports two input types:
        - List of words: List[List[str]]
        - List of sentences: List[List[List[str]]]
            - Each sentence is a list of words
            - Each word is a list of phonemes

        Example: 
        input_list = [
            [['p','u','ʁ'], ['l','ə'], ['ʁ','w','a','j','o','m']],  # sentence 1
            [['l','e'], ['ɡ','ɛ','ʁ'], ['d','ə']]                   # sentence 2
        ]
        """
         # Detect if input is list of sentences (list of list of list of str)
        if all(isinstance(sentence, list) and all(isinstance(word, list) for word in sentence) for sentence in input_list):
            flat_words = [word for sentence in input_list for word in sentence]
        else:
            flat_words = input_list  # assume already flat: list of words

        ngrams_per_word = []

        for word in flat_words: 
            if len(word) >= self.n: 
                word_ngrams  = list(ngrams(word, self.n))
                ngrams_per_word.extend(word_ngrams)
        logging.debug(f"ngram examples: {ngrams_per_word[:5]}")
        return ngrams_per_word

    def save_model(self, language, processing_type, text_type):
        """
        Save this specific model
        """
        os.makedirs(f"produced_data/{language}/{processing_type}", exist_ok=True)

        with open(f"produced_data/{language}/{processing_type}/{language}_{text_type}_markov_model_{self.n}gram.pkl", "wb") as f:
            pickle.dump(self, f)

        print(f"\n✅ Saved {self.n}-gram model to 'produced_data/{language}/{processing_type}/'")



    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
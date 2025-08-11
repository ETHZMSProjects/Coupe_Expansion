from nltk.util import ngrams
from collections import defaultdict
import os
import pickle
import warnings
import math
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
from info_rate import compute_info_rate
from helpers import validate_structure

logging.basicConfig(level=logging.INFO)


class BaseNgramModel: 
    def __init__(self, n): 
        """
        Initialize an n-gram Markov model.
        """
        self.n = n # n-gram order
        self.ngram_counts = defaultdict(int)  # n-gram counts
        self.prefix_counts = defaultdict(int)  # Counts how many times that exact prefix occurs, across all n-grams
        self.cond_probs = defaultdict(float)
        self.entropy_map = defaultdict(float) # entropy_map[prefix] holds H(Y|X=prefix)
        self.total_ngrams = 0
        self.total_prefixes = 0 # Total number of unique prefixes
        self.train_vocab = set() # training vocabulary (closed vocab)
    
    def build(self, set_type, data, text_type):
        """
        Build ngrams for within word transitions or sentence-level transitions at the level of the chosen linguistic unit.
        Optionally use pre-tuned lambdas_override for interpolation.
        """
        ngram_list = self.generate_ngrams(data, text_type)
        if not ngram_list:
            raise ValueError(
                f"No {self.n}-grams could be generated from the provided data using text_type='{text_type}")
        
        # Ensure a closed vocabulary by padding unknown tokens
        if not hasattr(self, "train_vocab"):
            self.train_vocab = set()
            self.train_vocab.add("<unk>")


        # Count each n-gram and prefix
        for gram in ngram_list:
            prefix, next_token = tuple(gram[:-1]), gram[-1]
            if set_type == "train":
                # Expand vocab from training data
                self.train_vocab.add(next_token)
                token_for_model = next_token
            else:
                # Map OOV tokens to <unk>
                token_for_model = next_token if next_token in self.train_vocab else "<unk>"

            self.ngram_counts[(prefix, token_for_model)] += 1
            self.prefix_counts[prefix] += 1
            self.total_ngrams += 1

        self.total_prefixes = len(self.prefix_counts)
    
    def generate_ngrams(self, data, text_type):
        if text_type == "across_sentences":  
            return self.generate_ngrams_for_sent(data)
        elif text_type == "within_words":
            return self.generate_ngrams_for_words(data)
        else: 
            warnings.warn("Unknown text_type. Use 'across_sentences' or 'within_words'.")
            return
    
    def generate_ngrams_for_sent(self, input_list):
        """
        Generate n-grams across words within each sentence, but not across sentences.
         Auto-detects if input is:
        - List[List[str]] where each str is a word → compute word-level n-grams
        - List[List[List[str]]] where each inner list is a word (sequence of phonemes/syllables) → compute unit-level n-grams
        
        input_list is a List of sentences:
        - each sentence is List of words
        - each word is a string or a list of phonemes or syllables
        
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
        # Detect input type by checking the first "word" in the first sentence
        first_word = input_list[0][0]
        is_word_string = isinstance(first_word, str)
        is_word_list = isinstance(first_word, list)

        ngrams_per_sentence = []

        for sentence in input_list:
            if is_word_string:
                # Sentence is list of words (strings) → word-level n-grams
                sentence_stream = sentence
            elif is_word_list:
                # Sentence is list of words (lists of units)  
                # Flatten sentence to stream of linguistic units
                sentence_stream = [segment for word in sentence for segment in word]
            else:
                raise ValueError("Unsupported input structure")
            
            # Generate n-grams if enough units
            if len(sentence_stream) >= self.n:
                sentence_ngrams = list(ngrams(sentence_stream, self.n))
                ngrams_per_sentence.extend(sentence_ngrams)

        #logging.info(f"📊 {self.n}-gram: {len(input_list)} sentences → {len(ngrams_per_sentence)} n-grams")
        #logging.info(f"ngram examples for across_sentences: {ngrams_per_sentence[:15]}")
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
        #logging.info(f"ngram examples for within_words: {ngrams_per_word[:15]}")

        return ngrams_per_word


    def save_model(self, language, folder, processing_type, text_type, corpus_size_str):
        """
        Save this specific model
        """
        os.makedirs(f"{folder}/{processing_type}", exist_ok=True)

        with open(f"{folder}/{processing_type}/{language}_{text_type}_markov_model_{self.n}gram_{corpus_size_str}.pkl", "wb") as f:
            pickle.dump(self, f)

        print(f"\n✅ Saved {self.n}-gram model to '{folder}/{processing_type}/'")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

class MLEModel(BaseNgramModel):
    def build(self, set_type, data, text_type):
        super().build(set_type, data, text_type)  # use base build to count
        
        # Compute MLE probabilities
        self.cond_probs = {
            (prefix, token): count / self.prefix_counts[prefix]
            for (prefix, token), count in self.ngram_counts.items()
        }

        # Verify probabilities sum to ~ 1
        for prefix in self.prefix_counts:
            total = sum(prob for (p, _), prob in self.cond_probs.items() if p == prefix)
            if abs(total - 1.0) > 1e-6:
                logging.warning(f"Probabilities for prefix {prefix} sum to {total:.4f}")

    def get_probability(self, prefix, token):
        return self.cond_probs.get((prefix, token), 0.0)
    

class JelinekMercerModel(MLEModel):
    def __init__(self, n: int, lambda_grid: Optional[List[float]] = None):
        super().__init__(n)
        # Default: smaller lambdas for higher orders (more backoff)
        self.grids = {
            1: [0.75, 0.9, 1.0],  # λ₁ — unigram
            2: [0.8, 0.9, 1.0],  # λ₂ — bigram # from Malagutti et al. (2024)
            3: [0.5, 0.6, 0.7],   # λ₃ — trigram
            4: [0.05, 0.1, 0.3, 0.5]    # λ₄ — 4-gram
        }

        self.p_jm_by_order = {}
        self.lambdas = {}
        self.models_by_order = {}  # stores all orders' MLE models

    def pre_compute_ngram_models(self, data, text_type):
        self.models_by_order = {}
        self.train_vocab = set()
        self.text_type = text_type

        for order in range(1, self.n + 1):
            model = MLEModel(order)
            model.build('train', data, text_type)
            self.models_by_order[order] = {
                'prefix_counts': model.prefix_counts,
                'ngram_counts': model.ngram_counts,
                'p_ml': model.cond_probs
            }
            self.train_vocab.update(model.train_vocab)


    def fit(self, data: List[str], text_type: str, lambdas: Dict[int, float]):
        """Fit JM model with given lambdas."""

        # builds MLE models on full dataset
        self.pre_compute_ngram_models(data, text_type)  

        logging.info(f"Training vocabulary size: {len(self.train_vocab)}")

        expected_keys = list(range(1, self.n+1))
        if not isinstance(lambdas, dict) or sorted(lambdas.keys()) != expected_keys:
            raise ValueError(f"Lambdas must be a dict with integer keys {expected_keys}")
        self.lambdas = lambdas

        self.p_jm_by_order = self.jm_interpolate(check_normalization=True)

    def train_dev_test_split_sentences(self, data, text_type, processing_type, dev_frac=0.02, test_frac=0.02, seed=42):
        """
        Split data into train, dev, and test sets, preserving original structure.

        Parameters
        ----------
        dev_frac : float
            Fraction of data for development (validation) set.
        test_frac : float
            Fraction of data for final held-out test set.
        """
        #logging.info(f"Data Preview: {data[:2]}")

        n_sent = len(data)
        rng = np.random.default_rng(seed)
        idx = np.arange(n_sent)
        rng.shuffle(idx)

        test_size = int(round(n_sent * test_frac))
        dev_size  = int(round(n_sent * dev_frac))

        test_idx = np.sort(idx[:test_size])
        dev_idx  = np.sort(idx[test_size:test_size + dev_size])
        train_idx = np.sort(idx[test_size + dev_size:])

        train = [data[i] for i in train_idx]
        dev   = [data[i] for i in dev_idx]
        test  = [data[i] for i in test_idx]

        # Validate structure of each split
        for split_name, split_data in zip(["Train", "Dev", "Test"], [train, dev, test]):
            validate_structure(split_data, text_type, processing_type, context=f"{split_name} Corpus")

        return train, dev, test



    def jm_interpolate(self, check_normalization=True) -> Dict[int, Dict]:
        """
        Perform Jelinek–Mercer (JM) smoothing via recursive interpolation.

        JM smoothing interpolates higher-order empirical (MLE) probabilities 
        with lower-order smoothed probabilities, applying the combination 
        recursively until a base case is reached.

        Formula:
            P_JM^(n)(w | h) = λ_n * P_ML^(n)(w | h) + (1 - λ_n) * P_JM^(n-1)(w | h')
            P_JM^(1)(w)     = λ_1 * P_ML^(1)(w)     + (1 - λ_1) * Uniform(w)

        Where:
        - P_ML^(n)(w | h) is the empirical MLE estimate from counts:
            P_ML^(n)(w | h) = count(h, w) / count(h)
        - P_JM^(n-1)(w | h') is the already-smoothed probability from the 
        (n-1)-gram model, with h' = h[1:] (history shortened by one token).
        - λ_n ∈ [0, 1] controls the weight given to the higher-order empirical 
        distribution vs. the lower-order smoothed distribution.
        - Uniform(w) = 1 / |V| is the zerogram (uniform) probability over the 
        vocabulary V, used here as the base case for unigrams.

        Implementation details in this method:
        1. Order 1 (unigrams) interpolates empirical unigram P_ML^(1) with a 
        uniform distribution over the vocabulary.
        2. Higher orders interpolate empirical P_ML^(n) with the recursively 
        smoothed (n-1)-gram probabilities from `p_jm_by_order`.
        3. Only observed continuations for a given prefix are stored; unseen 
        continuations receive their probability from the lower-order model 
        at query time.
        4. `check_normalization` optionally ensures that probability mass for 
        each prefix sums to ~1.0.

        This implementation is equivalent to the Jelinek–Mercer formulation 
        described in:
            Chen, S. F., & Goodman, J. (1999).
            An empirical study of smoothing techniques for language modeling. 
            Computer Speech & Language, 13(4), 359–394. 
            https://doi.org/10.1006/csla.1999.0128

        Returns
        -------
        Dict[int, Dict]
            A mapping from n-gram order to a dict of ((prefix), token) -> probability
            entries for the JM-smoothed model at that order.
        """
        if not isinstance(self.lambdas, dict) or len(self.lambdas) == 0:
            raise ValueError("Lambdas must contain at least one entry.")

        # Zerogram: uniform distribution
        uniform_prob = 1.0 / len(self.train_vocab)

        p_jm_by_order = {}
        max_order = max(self.lambdas.keys())

        # ----- Order 1: base case = pure ML for seen tokens, uniform for OOV -----
        p_ml_uni = self.models_by_order[1]['p_ml']
        p_jm_by_order[1] = {}
        for tok in self.train_vocab:
            ml_prob = p_ml_uni.get(((), tok), 0.0)
            p_jm_by_order[1][((), tok)] = self.lambdas[1] * ml_prob + (1 - self.lambdas[1]) * uniform_prob


        # ----- Orders >= 2: interpolate recursively -----
        for order in range(2, max_order + 1):
            current_ml = self.models_by_order[order]['p_ml']
            lower_jm = p_jm_by_order[order - 1]
            lambda_n = self.lambdas[order]

            new_p_jm = {}
            prefix_sums = defaultdict(float)

            # Iterate over the training vocabulary 
            for prefix in self.models_by_order[order]['prefix_counts']:
                lower_prefix = prefix[1:] if prefix else ()

                for tok in self.train_vocab:
                    # Compute interpolated probability
                    ml_prob = current_ml.get((prefix, tok), 0.0)
                    backoff_prob = lower_jm[(lower_prefix, tok)]
                    prob = lambda_n * ml_prob + (1 - lambda_n) * backoff_prob
                    # Store and track total mass
                    new_p_jm[(prefix, tok)] = prob
                    prefix_sums[prefix] += prob

                # Note: unseen continuations get prob from lower_jm automatically at query time

            p_jm_by_order[order] = new_p_jm

            # Optional: check normalization
            if check_normalization:
                for prefix, total in prefix_sums.items():
                    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                        logging.warning(f"Prefix {prefix} sums to {total:.6f}")

            
        return p_jm_by_order

    def get_probability(self, prefix: Tuple[str, ...], token: str) -> float:
        order = len(prefix) + 1
        uniform_prob = 1.0 / len(self.train_vocab) if self.train_vocab else 0.0

        while order >= 1:
            table = self.p_jm_by_order.get(order, {})
            candidate_prefix = prefix[-(order - 1):] if order > 1 else ()
            prob = table.get((candidate_prefix, token))
            if prob is not None:
                return prob
            order -= 1

        # If no probability found in any order, return uniform
        return uniform_prob
    

    def perplexity(self, set_type: str, data: List[str], text_type: str) -> float:
        """
        Calculate the perplexity of the model on a given dataset.

        Perplexity (PPL) is a standard measure of how well a probabilistic language model 
        predicts a sequence. It is related to cross-entropy (H) by PPL = 2 ** H, 
        where cross-entropy in bits per token is:

            H = - (1 / N) * Σ log₂ P(token | context)

        Interpretation:
        - PPL ≈ the model’s average “branching factor” for the next token.
        - Lower PPL → better predictions (more confident and accurate).
        - Higher PPL → worse predictions (more uncertainty).

        This method:
        1. Generates n-grams from `data` according to `text_type`.
        2. Sums the log₂ probabilities of each observed token given its prefix.
        3. Converts the average negative log₂ probability into perplexity.

        Parameters
        ----------
        data : List[str]
            Tokenized corpus; structure depends on `text_type`.
        text_type : str
            How n-grams are generated (e.g., 'across_sentences', 'within_words').

        Returns
        -------
        float
            Perplexity score for the dataset
        """

        if set_type != "train":
            data = self.map_oov_to_unk(data)

        ngram_list = self.generate_ngrams(data, text_type)
        total_logp = 0.0
        total_tokens = 0

        if not ngram_list:
            raise ValueError(
                f"No {self.n}-grams could be generated from the provided data using text_type='{text_type}'"
            )

        for gram in ngram_list:
            prefix = tuple(gram[:-1])
            token = gram[-1]
            prob = self.get_probability(prefix, token)
            total_logp += math.log2(prob if prob > 0 else 1e-10)  # bits
            total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        cross_entropy_bits = -total_logp / total_tokens
        return 2 ** cross_entropy_bits  # perplexity in bits

    def tune_lambdas(self, set_type, train_data: List[str], dev_data: List[str], text_type: str):
        """Grid search for best lambdas."""
        
        # Precompute MLE models for all orders for train_data
        self.pre_compute_ngram_models(train_data, text_type)
        best_lambdas = {}
        best_ppl_overall = None 
        
        for order in range(1, self.n + 1):

            best_lambda = None
            best_ppl = float('inf')
            grid = self.grids[order]

            for lam in grid:
                test_lambdas = {o: best_lambdas[o] for o in range(1, order)}
                test_lambdas[order] = lam

                self.lambdas = test_lambdas
                self.p_jm_by_order = self.jm_interpolate(check_normalization=False)
                ppl = self.perplexity(set_type, dev_data, text_type)

                if ppl < best_ppl:
                    best_ppl = ppl
                    best_lambda = lam

            logging.info(
                f"[Tune for {set_type}] λ_{order} = {best_lambda} | dev PPL={best_ppl:.3f}"
            )

            best_lambdas[order] = best_lambda

        # Fit final JM model with tuned lambdas
        self.lambdas = best_lambdas
        self.p_jm_by_order = self.jm_interpolate(check_normalization=True)
        best_ppl_overall = self.perplexity(set_type, dev_data, text_type)
        
        return best_lambdas, best_ppl_overall
        
    def compute_conditional_entropy(self, use_jm=True):
        """
        Compute conditional entropy H(Y|X) = Σ_x P(x) * H(Y|X=x)
        Note that while P(y|x) is smoothed, P(x) uses empirical counts 

        Parameters
        ----------
        use_jm : bool
            If True, use JM-smoothed probabilities (p_jm_by_order).
            If False, use raw MLE probabilities (p_ml from models_by_order).
        """
        order = self.n
        if use_jm:
            cond_probs = self.p_jm_by_order[order]
        else:
            cond_probs = self.models_by_order[order]['p_ml']

        prefix_counts = self.models_by_order[order]['prefix_counts']

        # Compute H(Y|X=x) for each prefix
        prefix_entropy = defaultdict(float)
        for (prefix, token), prob in cond_probs.items():
            if prob > 0:
                prefix_entropy[prefix] += -prob * math.log2(prob)

        # Weight each prefix's entropy by empirical P(x) from corpus
        total_prefix_count = sum(prefix_counts.values())
        total_entropy = sum(
            (prefix_counts[prefix] / total_prefix_count) * cond_entropy
            for prefix, cond_entropy in prefix_entropy.items()
        )

        return total_entropy

    

    def fit_with_tuning(self, data, text_type, processing_type, language, dev_frac=0.02, test_frac=0.02, seed=42):
        """
        Tune lambdas on dev set, retrain on train+dev, evaluate on test set.
        """
        # Check nesting 
        try:
            validate_structure(data, text_type, processing_type, context="Corpus")
        except (TypeError, ValueError) as e:
            logging.error(f"Data structure invalid before {text_type} split: {e}")
            raise

        # 1. Split data
        train_data, dev_data, test_data = self.train_dev_test_split_sentences(
            data, text_type, processing_type, dev_frac=dev_frac, test_frac=test_frac, seed=seed
        )

        logging.info(
            f"[Split] {text_type} | total={len(data)} sentences → "
            f"train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}"
        )

        # 2. 2. Tune lambdas on dev (train only on train_data) (this already builds MLE models)
        best_lambdas, dev_ppl = self.tune_lambdas('train', train_data, dev_data, text_type)
        if best_lambdas is None:
            raise ValueError(f"No best lambdas found for n={self.n}.")
        
        logging.info(f"Best λs for n={self.n}: {best_lambdas} (dev ppl={dev_ppl:.3f})")

        # 3. Retrain on train+dev
        train_plus_dev = train_data + dev_data
        self.fit(train_plus_dev, text_type, best_lambdas)

        # 4. Compute Information Density
        info_density = self.compute_conditional_entropy(use_jm=True)

        # 5. Evaluate on test set
        test_ppl = self.perplexity('test', test_data, text_type)
        logging.info(f"[Final Eval] Best λs: {best_lambdas} |Test PPL={test_ppl:.3f} | {text_type}")    

        # 6. Compute Information Rate
        info_rate_values, speech_rate_values = compute_info_rate(
            info_density, processing_type, language
        )

        # 7. Save tuned lambdas
        lambda_values = [best_lambdas[i] for i in range(1, self.n + 1)]
        with open(f"best_lambdas_n{self.n}.json", "w") as f:
            json.dump({
                "lambdas_dict": best_lambdas,
                "lambdas_list": lambda_values,
                "n": self.n,
                "dev_ppl": dev_ppl
            }, f, indent=2)

        return {
            'model': self,
            "best_lambdas": best_lambdas,
            'test_perplexity': test_ppl,
            "dev_perplexity": dev_ppl,
            "info_density": info_density,
            "info_rate_values": info_rate_values,
            "speech_rate_values": speech_rate_values
        }
    
    def map_oov_to_unk(self, data):
        """
        Replace tokens not in training vocab with <unk>.
        Works for both 'within_words' and 'across_sentences' formats.
        """
        mapped = []

        for sentence in data:
            mapped_sentence = []
            for word in sentence:
                if isinstance(word, str):
                    # word-level token (across_sentences word-level)
                    tok = word if word in self.train_vocab else '<unk>'
                    mapped_sentence.append(tok)
                elif isinstance(word, list):
                    # list of units (phonemes/syllables)
                    mapped_word = [
                        tok if tok in self.train_vocab else '<unk>'
                        for tok in word
                    ]
                    mapped_sentence.append(mapped_word)
                else:
                    raise ValueError(f"Unsupported token structure: {word}")
            mapped.append(mapped_sentence)

        return mapped



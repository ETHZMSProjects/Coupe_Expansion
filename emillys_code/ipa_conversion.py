import string
import subprocess
import unicodedata
import regex as re
import torch
import unicodedata
import jieba
import logging
import unicodedata
from more_itertools import chunked
from transformers import T5ForConditionalGeneration, AutoTokenizer
from functools import partial
from joblib import Memory
from collections import defaultdict
from process_ipa import merge_clitics, convert_numbers
import os
from tqdm import tqdm
from pathlib import Path


# ensure that espeak-ng is discoverable for all subprocesses during that session.
os.environ["PATH"] = "/home/emilly/.local/bin:" + os.environ["PATH"]

# Caching (optional)
memory = Memory("cache_dir", verbose=1)
@memory.cache
def get_ipa_espeak_cached(word, espeak_code):
    return get_ipa_espeak(word, espeak_code)

# --- CharsiuG2P Model Setup ---
# Load model and tokenizer once to avoid repeated loading
# Choose different model sizes (e.g., 'charsiu/g2p_multilingual_byT5_small_100')

CHARSIU_MODEL_NAME = 'charsiu/g2p_multilingual_byT5_small_100' #'charsiu/g2p_multilingual_byT5_tiny_16_layers_100' # load better model when not testing
charsiu_tokenizer = None
charsiu_model = None

def load_charsiu_model():
    """Loads the CharsiuG2P model and tokenizer, or retrieves them if already loaded."""
    global charsiu_tokenizer, charsiu_model
    if charsiu_model is None or charsiu_tokenizer is None:
        charsiu_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
        charsiu_model = T5ForConditionalGeneration.from_pretrained(CHARSIU_MODEL_NAME)
        """if torch.cuda.is_available():
            charsiu_model.to('cuda')
            logging.info(f"{CHARSIU_MODEL_NAME} moved to GPU.")
        logging.info(f"{CHARSIU_MODEL_NAME} running on CPU.")"""
    return charsiu_tokenizer, charsiu_model


def parallelize_ipa_generation(text, language, tokenizer, model, config_dict):
    """
    Flattens sentences into words, batch-generates IPA using G2P, 
    and reconstructs the sentence structure.
    
    Args:
        text (list of list of str): Sentences as lists of words.
        language (str): Language code (e.g., "FRA").
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model.

    Returns:
        list of list of str: Sentences as lists of IPA transcriptions.
    """
    # Normalize and flatten
    text = [[unicodedata.normalize("NFKC", word) for word in sentence]  for sentence in text]

    sentences_cleaned = []

    if language in ["FRA", "ENG"]:
        merge = partial(merge_clitics, language=language)
    else:
        merge = lambda x: x # do nothing
    
    flat_words_tagged = []  # (word, sentence_index)
    for i, sentence in enumerate(text):
        sentence = merge(sentence)

        # Remove list entries that are only punctuation or whitespace
        cleaned = [word for word in sentence if any(char.isalnum() for char in word)]
        if cleaned:
            sentences_cleaned.append(cleaned)
            # Keep track of the word index in the original sentence
            flat_words_tagged.extend([(word, i) for word in cleaned])
        else: 
            logging.warning(f"Skipped sentence {i}: {sentence}")

    # Batch IPA generation
    ipa_flat, updated_word_list, updated_sentence_ids = generate_ipa(flat_words_tagged, language, tokenizer, model, config_dict)

    # Reconstruct sentence structure
    sentence_map = defaultdict(list)
    for ipa, sid in zip(ipa_flat, updated_sentence_ids):
        sentence_map[sid].append(ipa)

    ipa_sentences = [sentence_map[i] for i in range(len(sentences_cleaned))]
    for orig, ipa in zip(text, ipa_sentences):
        logging.debug(f"{orig} → {ipa}")

    return ipa_sentences


def generate_ipa(word_list_tagged, language, tokenizer, model, config_dict):

    espeak = True if language in ['ENG', 'FRA', 'CMN', 'DEU', 'ITA', 'ESP'] else False
        
    if not word_list_tagged:
        return []
    
    word_list, sentence_ids = zip(*word_list_tagged)  # both are tuples
    word_list_original = list(word_list)
    sentence_ids = list(sentence_ids)

     # Convert numbers to words
    word_list = convert_numbers(word_list_original, language, config_dict)
    
    if language in ['CMN']:
            # Tokenize each item in the word list using jieba
            tokenized = [list(jieba.cut(word, cut_all=False)) for word in word_list]
            word_list = [tok for sublist in tokenized for tok in sublist]

            # Expand sentence_ids to match tokenized list
            sentence_ids = [
                sid for tokens, sid in zip(tokenized, sentence_ids)
                for _ in tokens
            ]
            logging.debug(f"jierba: {word_list}")

    ###  Use espeak to get IPA
    if espeak: 
        espeak_code = config_dict['espeak Code']

        # Chunks the word list into batches for subprocess efficiency
        word_list = [unicodedata.normalize("NFC", w) for w in word_list]
        word_chunks = list(chunked(word_list, 128))

        results_nested = [get_espeak_ipa_batch(chunk, espeak_code) 
                  for chunk in tqdm(word_chunks, desc=f"🔠 Converting {language} corpus to IPA using espeak", unit="batch")]

        raw_ipa = [ipa for chunk in results_nested for ipa in chunk]

        #for word, ipa in zip(word_list, raw_ipa):
            #if not ipa:
                #print(f"🧪 EMPTY: '{word}' → '{ipa}'")

        clean = partial(clean_ipa, as_string=True, delimiter='', config_dict=config_dict)
        ipa_results = [clean(ipa) for ipa in raw_ipa]
    
    else: 
         ### Use g2p model to generate IPA

        # CharsiuG2P requires a language prefix and a space after the colon
        # Example: "<eng>: hello" or "<fra>: bonjour"
        charsiu_code = config_dict['charsiu Code']
        tagged_words = [f"<{charsiu_code}>: {word.lower()}" for word in word_list] 
        ipa_results = []
        model.eval()

        # Move model to GPU once if available
        if torch.cuda.is_available():
            model = model.to('cuda')

        with torch.no_grad():
            for batch in tqdm(chunked(tagged_words, 128), desc=f"🔠 Converting {language} corpus to IPA using CharsiuG2P", unit="batch"):
                encoded = tokenizer(batch, padding=True, return_tensors='pt')
                if torch.cuda.is_available():
                    encoded = {k: v.to('cuda') for k, v in encoded.items()}
                preds = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    num_beams=1,
                    max_length=50,
                    do_sample=False
                )
                decoded = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
                cleaned_batch = [clean_ipa(ipa, True, '') for ipa in decoded]
                ipa_results.extend([ipa for ipa in cleaned_batch if ipa])

    # filter out empty entries in both the original and the ipa word list
    filtered = [(w, ipa, sid) for w, ipa, sid in zip(word_list, ipa_results, sentence_ids) if ipa]
    ipa_list = [ipa for _, ipa, _ in filtered]
    updated_word_list = [w for w, _, _ in filtered]
    updated_sentence_ids = [sid for _, _, sid in filtered]
    return ipa_list, updated_word_list, updated_sentence_ids



# Regex for grapheme clusters
GRAPHEME_RE = re.compile(r'\X', re.UNICODE)

def clean_ipa(ipa_string, as_string, delimiter, config_dict):
    """
    Cleans a given IPA string by removing non-phonemic characters,
    while preserving delimiter and language-specific meaningful IPA symbols.
    """
    keep_chars = set(config_dict["Keep Characters"]) 

    if isinstance(ipa_string, list):
        ipa_string = " ".join(ipa_string)
    
    # First, globally remove any ( ... ) artifacts
    ipa_string = re.sub(r"\(.*?\)", "", ipa_string)

    segments = GRAPHEME_RE.findall(ipa_string)
    segments = [re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', seg) for seg in segments]

    # Preserve any characters in the delimiter
    # Split delimiters (e.g., '[.-]' → {'.', '-'})
    delimiter_chars = set()
    if isinstance(delimiter, str):
        if delimiter.startswith("[") and delimiter.endswith("]"):
            delimiter_chars = set(delimiter[1:-1])
        else:
            delimiter_chars = set(delimiter)

     # Full preservation set
    PRESERVE = keep_chars | delimiter_chars

    STRIP_CHARS = {
        'ˈ', 'ˌ', '.', ',', '-', '/', '!', '?', ';', ' ', '-', '(', ')', '"', "'", '`', '’',
        '“', '”', '‘', '’', '《', '》', '【', '】','[', ']', '{', '}', '§', '%', ' ',
        '&', '#', '@', '…', '—', '–', '～', '·', '「', '」', '『', '』', '_', '=', '+', '*', '^', '~',
        '\n', '\t', '\r', '"', "'", '’', '`', '。', '、', '，', '！', '？', '；', '：' , 'ʼ', 'ʹ', 'ʽ'
    } - PRESERVE  # Subtract preserved symbols

    STRIP_PATTERN = re.compile("|".join(map(re.escape, STRIP_CHARS)))

    # Clean the segments
    cleaned_as_list = []
    for seg in segments:
        seg = STRIP_PATTERN.sub("", seg) 
        if any(unicodedata.category(char).startswith('S') for char in seg):  # Symbol characters
            continue
        seg = seg.strip() # remove trailing spaces and empty segments
        if seg:
            cleaned_as_list.append(seg)
    if not cleaned_as_list: 
        return None
        
    return "".join(cleaned_as_list) if as_string else cleaned_as_list

def get_espeak_ipa_batch(chunk, espeak_code):
    """Run espeak-ng on a batch of words, one at a time."""
    return [get_ipa_espeak(word, espeak_code) for word in chunk]

def get_ipa_espeak(word, espeak_code):
    """Get IPA transcription from espeak-ng."""

    word = unicodedata.normalize("NFKC", word)
    word = word.strip(string.punctuation + "’‘“”").strip().lower()

    if not word:
        return ""

    try:
        result = subprocess.run(
            ['espeak-ng', '-v', espeak_code, '--ipa=3', '-q', word.lower()],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            #logging.error(f"❌ espeak-ng returned error for '{word}': {result.stderr.strip()}")
            print(f"❌ espeak-ng returned error for '{word}': {result.stderr.strip()}")
            return ""

        ipa = result.stdout.strip()
        if not ipa:
            #logging.warning(f"⚠️ espeak-ng gave empty output for '{word}'")
            #print(f"⚠️ espeak-ng gave empty output for '{word}'")
            return ""

        return ipa

    except Exception as e:
        logging.error(f"❌ espeak-ng crashed on word '{word}': {e}")
        return ""
    

def get_specific_ipa_corpus(language, desired_size, folder): 
    folder = Path(folder)
    if not folder.exists():
        return False, None

    for file in folder.glob(f"ipa_corpus_{language}_size:*.pkl"):
        size = int(file.stem.split("_size:")[-1])
        if size == desired_size:
            return True, file

    return False, None

def get_largest_ipa_corpus(language, expected_size, folder, tolerance=1000):
    folder = Path(folder)
    if not folder.exists():
        return False, False, None

    candidates = []
    for file in folder.glob(f"ipa_corpus_{language}_size:*.pkl"):
        size = int(file.stem.split("_size:")[-1])
        candidates.append((size, file))

    if not candidates:
        return False, False, None

    max_size, best_file = max(candidates)
    is_near_expected = abs(max_size - expected_size) <= tolerance
    return True, is_near_expected, best_file
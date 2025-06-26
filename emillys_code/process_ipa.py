import pandas as pd
import re
import string
import subprocess
import unicodedata
import unicodedata
import regex as re
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from pathlib import Path
from num2words import num2words
from more_itertools import chunked
from tqdm import tqdm
from itertools import accumulate


def load_config(language, key):
    config_path = Path("C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/language_config.json")
    config_df = pd.read_json(config_path)
    config_df.set_index("Language", inplace=True)

    try:
        lang_cfg = config_df.loc[language]
        result = lang_cfg[key]
    except KeyError:
        print(f"Key '{key}' not found, either {language} or {key} is not supported.")
        return
    return result


def merge_clitics(tokens):
    merged = []
    skip = False
    for i, token in enumerate(tokens):
        if skip:
            skip = False
            continue
        if token.endswith("'") and i + 1 < len(tokens):
            merged.append(token + tokens[i + 1])
            skip = True
        else:
            merged.append(token)
    return merged


# --- CharsiuG2P Model Setup ---
# Load model and tokenizer once to avoid repeated loading
# Choose different model sizes (e.g., 'charsiu/g2p_multilingual_byT5_small_100')
# Use the 'tiny' model for testing.
CHARSIU_MODEL_NAME = 'charsiu/g2p_multilingual_byT5_tiny_16_layers_100'
charsiu_tokenizer = None
charsiu_model = None

def load_charsiu_model():
    """Loads the CharsiuG2P model and tokenizer, or retrieves them if already loaded."""
    global charsiu_tokenizer, charsiu_model
    if charsiu_model is None or charsiu_tokenizer is None:
        print(f"Loading CharsiuG2P model: {CHARSIU_MODEL_NAME}...")
        charsiu_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
        charsiu_model = T5ForConditionalGeneration.from_pretrained(CHARSIU_MODEL_NAME)
        # Move model to GPU if available
        if torch.cuda.is_available():
            charsiu_model.to('cuda')
            print("CharsiuG2P model moved to GPU.")
        else:
            print("CharsiuG2P model running on CPU.")
    return charsiu_tokenizer, charsiu_model


def convert_numbers(word, language): 
    num2word_code = load_config(language, "num2words Code")
    if language not in ["YUE", "CMN", "VIE"]: 
        # convert numbers to words
        try:
            word = re.sub(r'\d+', lambda x: num2words(int(x.group()), lang=num2word_code.lower()), word)
        except NotImplementedError:
            print(f"Language '{language}' not supported by num2words. Skipping number conversion.")
    return word

def parallelize_ipa_generation(text, language, tokenizer, model):
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
    
    # Remove punctuation tokens and flatten
    punctuation = {'.', ',', '?', '!'}
    text = [[word for word in sentence if word not in punctuation] for sentence in text]

    flat_words = []
    sentence_lengths = []

    for sentence in text:
        if language == "FRA":
            sentence = merge_clitics(sentence)
        sentence_lengths.append(len(sentence))
        flat_words.extend(sentence)

    print("🔠 Generating IPA...")

    # Batch IPA generation
    ipa_flat = generate_ipa(flat_words, language, tokenizer, model)

    # Sanity check
    if not all(ipa_flat):
        for w, ipa in zip(flat_words, ipa_flat):
            if not ipa:
                print(f"⚠️ Failed to generate IPA for word: {w}")
        return [], []

    # Reconstruct sentence structure
    indices = list(accumulate(sentence_lengths))
    start = 0
    ipa_sentences = []
    for end in indices:
        ipa_sentences.append(ipa_flat[start:end])
        start = end

    return ipa_sentences


def generate_ipa(word_list, language, tokenizer, model, stressed = False):
    if not word_list:
        return []
    
    # Convert numbers
    word_list = [convert_numbers(w, language) for w in word_list]
    
    ### 1. Use g2p model to generate IPA

    # CharsiuG2P requires a language prefix and a space after the colon
    # Example: "<eng>: hello" or "<fra>: bonjour"
    charsiu_code = load_config(language, 'charsiu Code')
    tagged_words = [f"<{charsiu_code}>: {w.lower()}" for w in word_list]

    ipa_results = []
    model.eval()

    # Move model to GPU once if available
    if torch.cuda.is_available():
        model = model.to('cuda')

    with torch.no_grad():
        for batch in chunked(tagged_words, 64):
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
            cleaned_batch = [clean_ipa(ipa, True, '', language) for ipa in decoded]
            ipa_results.extend([ipa for ipa in cleaned_batch if ipa])

    ### 2. Use espeak to get IPA with stress marks
    if stressed: 
        espeak_code = load_config(language, 'espeak Code')
        ipa_results = [get_ipa_espeak(w, espeak_code) for w in word_list]

    return ipa_results


# Regex for grapheme clusters
GRAPHEME_RE = re.compile(r'\X', re.UNICODE)

def clean_ipa(ipa_string, as_string, delimiter, language):
    """
    Cleans a given IPA string by removing non-phonemic characters,
    while preserving delimiter and language-specific meaningful IPA symbols.
    """

    # Define language-specific meaningful symbols to preserve
    config_df = pd.read_json("C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/language_config.json")
    config_df.set_index("Language", inplace=True)
    lang_cfg = config_df.loc[language]
    keep_chars = set(lang_cfg["Keep Characters"])
    
    if isinstance(ipa_string, list):
        ipa_string = " ".join(ipa_string)
    
    # First, globally remove any ( ... ) artifacts
    ipa_string = re.sub(r"\(.*?\)", "", ipa_string)

    segments = GRAPHEME_RE.findall(ipa_string)

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
        '\n', '\t', '\r', '"', "'", '’', '`', '。', '、', '，', '！', '？', '；', '：'
    } - PRESERVE  # Subtract preserved symbols

    # Clean the segments
    cleaned_as_list = []
    for seg in segments:
        seg = re.sub(r"\(.*?\)", "", seg)  # remove weird parenthesis artifacts
        if seg in STRIP_CHARS:
            continue
        if any(unicodedata.category(char).startswith('S') for char in seg):  # Symbol characters
            continue
        seg = seg.strip() # remove trailing spaces and empty segments
        if seg:
            cleaned_as_list.append(seg)
    if not cleaned_as_list: 
        return None
        
    return "".join(cleaned_as_list) if as_string else cleaned_as_list


def get_ipa_espeak(word, espeak_code):
    """Get IPA transcription from espeak-ng."""

    # normalize the word
    word = unicodedata.normalize("NFC", word)
    # remove punctuation
    no_punct = word.strip(string.punctuation)

    try:
        result = subprocess.run(
            ['espeak-ng', '-v', espeak_code, '--ipa=3', '-q', no_punct.lower()],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"espeak-ng failed on word '{word}': {e}")
        return ""
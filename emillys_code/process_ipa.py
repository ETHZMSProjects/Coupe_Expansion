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



def generate_ipa(orthographic_word, language, tokenizer, model):
    if not orthographic_word:
        return "", ""
    
    ### 1. Use g2p model to generate IPA

    # CharsiuG2P requires a language prefix and a space after the colon
    # Example: "<eng>: hello" or "<fra>: bonjour"
    charsiu_code = load_config(language, 'charsiu Code')
    input_text = f"<{charsiu_code}>: {orthographic_word}"

    try:
        input_ids = tokenizer(
            [input_text],
            padding=True,
            add_special_tokens=False,
            return_tensors='pt'
        ).input_ids

        # Move input to GPU if model is on GPU
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        # Generate phonetic transcription in ipa
        preds = model.generate(
            input_ids,
            num_beams=1, # greedy decoding
            max_length=50, # Adjust max_length as needed for longer words
            do_sample=False # For deterministic output
        )

        # Decode the generated tokens
        g2p_ipa = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)[0]

        cleaned_g2p_ipa = clean_ipa(g2p_ipa, True, '', language)  # Clean the IPA output (mostly not necessary for CharsiuG2P)

        ### 2. Use espeak to get IPA with stress marks
        espeak_code = load_config(language, 'espeak Code')
        stressed_ipa = get_ipa_espeak(orthographic_word, espeak_code)

        return cleaned_g2p_ipa, stressed_ipa

    except Exception as e:
        print(f"Error generating ipa for '{orthographic_word}' in '{language}': {e}")
        return "", "" # Return empty string on error
    


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
        '\n', '\t', '\r', '"', "'", '’', '`', '。', '、', '，', '！', '？', '；', '：',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
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
            ['espeak-ng', '-v', espeak_code, '--ipa=3', '-q', no_punct],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"espeak-ng failed on word '{word}': {e}")
        return ""
import pandas as pd
import re
import string
import subprocess
import unicodedata
import unicodedata
import regex as re
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from num2words import num2words
from more_itertools import chunked
from config_loader import load_config
import unicodedata
import jieba
import logging
from joblib import Parallel, delayed, Memory
from more_itertools import chunked

jieba.setLogLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

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
        # Move model to GPU if available
        if torch.cuda.is_available():
            charsiu_model.to('cuda')
            logging.info(f"{CHARSIU_MODEL_NAME} moved to GPU.")
        else:
            logging.info(f"{CHARSIU_MODEL_NAME} running on CPU.")
    return charsiu_tokenizer, charsiu_model


def convert_numbers(word, language): 
    num2word_code = load_config(language, "num2words Code")
    if language not in ["YUE", "CMN", "VIE"]: 
        # convert numbers to words
        try:
            word = re.sub(r'\d+', lambda x: num2words(int(x.group()), lang=num2word_code), word)
        except NotImplementedError:
            logging.warning(f"Language '{language}' not supported by num2words. Skipping number conversion.")
    return word


def merge_clitics(tokens, language):
    if language == "FRA":
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
    else:
        return tokens 
    
def merge_diphthongs(phones):
    """
    Merges diphthongs and triphthongs in a list of IPA phones into single tokens for syllabification.
    Handles English, German, and French diphthongs/triphthongs with proper prioritization.
    
    Prioritization:
    1. Triphthongs (3 phones) first
    2. Core diphthongs (vowel + vowel) over glide + vowel
    3. Glide + vowel combinations
    
    Args:
        phones (List[str]): A list of phone-level IPA.
    
    Returns:
        List[str]: A new list with diphthongs/triphthongs merged as single items.
    """
    
    # TRIPHTHONGS: Vowel + Glide + Schwa/Rhotic (highest priority)
    triphthongs = {
        # English triphthongs
        'aɪə', 'aʊə', 'eɪə', 'oʊə', 'ɔɪə',
        'aɪɚ', 'aʊɚ', 'eɪɚ', 'oʊɚ', 'ɔɪɚ',
        # English alternative realizations
        'juə', 'jʊə', 'jɪə',
        # German (e.g. poetic or dialectal)
        'aɪə', 'aʊə', 'ɔɪə',
        # French (in gliding speech)
        'waɪ', 'waj', 'ɥij', 'ɥiə'
    }

    # CORE DIPHTHONGS: Vowel + Vowel combinations (second priority)
    core_diphthongs = {
        # English
        'eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'oʊ',
        'ɪə', 'ɛə', 'ʊə', 'ɑə', 'ɔə',
        'iə', 'uə', 'eə', 'əʊ',
        'ɪɚ', 'ɛɚ', 'ʊɚ', 'ɔɚ', 'aɚ', 'ɚə',

        # German
        'aɪ', 'aʊ', 'ɔɪ',
        'iə', 'eə', 'uə', 'oə', 'øə', 'yə', 'ɔə', 'ɛə', 'ɪə',
        'øy', 'œy',

        # French
        'ei', 'ɛi', 'ɔi', 'ui', 'øi', 'ie', 'ye', 'ue',
        'au', 'eu', 'ɛu', 'ou', 'ɔu', 'œu', 'iu', 'io',
        'iə', 'uə', 'eə', 'oə', 'ɑə',

        # Common vowel-vowel across languages
        'ɪi', 'ʊu', 'ɛe', 'ɔo', 'aə'
    }

    # GLIDE + VOWEL DIPHTHONGS: j/w/ɥ + Vowel (third priority)
    glide_vowel_diphthongs = {
        'ja', 'je', 'ji', 'jo', 'ju', 'jɑ', 'jɛ', 'jɪ', 'jɔ', 'jʊ', 'jə', 'jɚ',
        'wa', 'we', 'wi', 'wo', 'wu', 'wɑ', 'wɛ', 'wɪ', 'wɔ', 'wʊ', 'wə', 'wɚ',
        'ɥa', 'ɥe', 'ɥi', 'ɥo', 'ɥu', 'ɥy', 'ɥø', 'ɥœ', 'ɥɛ', 'ɥɔ', 'ɥɑ',
        'jy', 'jø', 'jœ', 'wy', 'wø', 'wœ'
    }

    # VOWEL + GLIDE DIPHTHONGS: Vowel + j/w (lowest priority)
    vowel_glide_diphthongs = {
        'aj', 'ej', 'ij', 'oj', 'uj', 'ɑj', 'ɛj', 'ɪj', 'ɔj', 'ʊj', 'əj', 'ɚj',
        'aw', 'ew', 'iw', 'ow', 'uw', 'ɑw', 'ɛw', 'ɪw', 'ɔw', 'ʊw', 'əw', 'ɚw',
        'øyj', 'œyj', 'øyw', 'œyw'
    }

    
    merged = []
    i = 0
    
    while i < len(phones):
        matched = False

        # Attach markers to previous segment
        if phones[i] in {'ː', 'ˑ'} and merged:
            merged[-1] += phones[i]
            i += 1
            continue
        
        # Check triphthongs first (highest priority)
        if i + 2 < len(phones):
            candidate = phones[i] + phones[i + 1] + phones[i + 2]
            if candidate in triphthongs:
                merged.append(candidate)
                i += 3
                matched = True
                continue
        
        # Before merging any diphthong, check if there's a triphthong starting at i+1
        # that would be broken by merging a diphthong at i
        if not matched and i + 1 < len(phones):
            candidate_diphthong = phones[i] + phones[i + 1]
            
            # Check if merging this diphthong would prevent a triphthong at i+1
            should_skip_for_triphthong = False
            if i + 3 < len(phones):
                potential_triphthong = phones[i + 1] + phones[i + 2] + phones[i + 3]
                if potential_triphthong in triphthongs:
                    should_skip_for_triphthong = True
            
            if not should_skip_for_triphthong:
                # Priority 1: Core diphthongs (vowel + vowel)
                if candidate_diphthong in core_diphthongs:
                    merged.append(candidate_diphthong)
                    i += 2
                    matched = True
                # Priority 2: Glide + vowel
                elif candidate_diphthong in glide_vowel_diphthongs:
                    merged.append(candidate_diphthong)
                    i += 2
                    matched = True
                # Priority 3: Vowel + glide
                elif candidate_diphthong in vowel_glide_diphthongs:
                    merged.append(candidate_diphthong)
                    i += 2
                    matched = True
        
        # If no match found, keep the single phone
        if not matched:
            merged.append(phones[i])
            i += 1
    
    return merged
    

def merge_diphthongs_post(word):
    """
    Merges diphthongs that have been incorrectly split across two adjacent syllables.

    This function checks for known diphthongs that may have been split at syllable
    boundaries — for example, ['ma', 'ɪ'] → ['maɪ']. It looks at the final character 
    of one syllable and the first character of the next, and if they form a known 
    diphthong, it merges the two syllables accordingly.

    Args:
        word (list of str): A word represented as a list of syllables,
                            where each syllable is a string of phonemes.

    Returns:
        list of str: The corrected list of syllables, with diphthongs merged
                     where appropriate.

    """
    diphthongs = {'aɪ', 'eɪ', 'ɔɪ', 'aʊ', 'əʊ', 'oʊ', 'ɪə', 'eə', 'ʊə'}
    fixed_word = []
    i = 0
    while i < len(word):
        if i + 1 < len(word):
            left = word[i]
            right = word[i + 1]
            # look at the last char of left and first char of right
            possible_diphthong = left[-1] + right[0]
            if possible_diphthong in diphthongs:
                merged = left[:-1] + possible_diphthong + right[1:]
                fixed_word.append(merged)
                logging.debug(f"merged diphtong: {merged}")
                i += 2
                continue
        fixed_word.append(word[i])
        i += 1
    return fixed_word



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
    
    # Remove punctuation tokens, normalize and flatten
    punctuation = {'.', ',', '?', '!'}
    text = [[word for word in sentence if word not in punctuation] for sentence in text]
    text = [[unicodedata.normalize("NFKC", word) for word in sentence]  for sentence in text]

    flat_words = []
    sentence_lengths = []

    for sentence in text:
        if language in ["FRA", "ENG"]:
            sentence = merge_clitics(sentence, language)
        sentence_lengths.append(len(sentence))
        flat_words.extend(sentence)

    print("🔠 Converting to IPA...")

    # Batch IPA generation
    ipa_flat = generate_ipa(flat_words, language, tokenizer, model)

    # Sanity check
    if not all(ipa_flat):
        for w, ipa in zip(flat_words, ipa_flat):
            if not ipa:
                logging.warning(f"⚠️ Failed to generate IPA for word: {w}")
        return [], []

    # Reconstruct sentence structure
    indices = list(pd.Series(sentence_lengths).cumsum())
    start = 0
    ipa_sentences = []
    for end in indices:
        ipa_sentences.append(ipa_flat[start:end])
        start = end

    #for orig, ipa in zip(text, ipa_sentences):
        #logging.debug(f"{orig} → {ipa}")

    return ipa_sentences


def generate_ipa(word_list, language, tokenizer, model):

    espeak = True if language in ['ENG', 'FRA', 'CMN', 'DEU', 'ITA', 'ESP'] else False
        
    if not word_list:
        return []
    
    # Convert numbers
    word_list = [convert_numbers(word, language) for word in word_list]
    word_list = [word for word in word_list if word.strip() not in {"'", "’", "", ":", "。", "?","¿", "...", ":", ";", "«", "»", "-", "–","“", "„","%", "/"}]
    
    if language in ['CMN']:
            word_list = [list(jieba.cut(sentence, cut_all=False)) for sentence in word_list]
            logging.debug(f"jierba: {word_list}")

    ###  Use espeak to get IPA
    if espeak: 
        espeak_code = load_config(language, 'espeak Code')
        raw_ipa = Parallel(n_jobs=-1)(
            delayed(get_ipa_espeak_cached)(w, espeak_code) for w in word_list
        )
        ipa_results = [clean_ipa(ipa, True, '', language) for ipa in raw_ipa]
    
    else: 
         ### Use g2p model to generate IPA

        # CharsiuG2P requires a language prefix and a space after the colon
        # Example: "<eng>: hello" or "<fra>: bonjour"
        charsiu_code = load_config(language, 'charsiu Code')
        tagged_words = [f"<{charsiu_code}>: {word.lower()}" for word in word_list] 
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
        logging.error(f"espeak-ng failed on word '{word}': {e}")
        return ""
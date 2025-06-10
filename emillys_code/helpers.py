from math import log2
import pandas as pd
import warnings
import os
import re
from pathlib import Path
import subprocess
import numpy as np
import jieba
import unicodedata
import regex as re
import pycantonese
from g2pk import G2p
from pypinyin import Style, pinyin as pypinyin_fn
import string


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


def update_values_in_csv(language_to_update, value, n, value_type):
    model = {1: "unigram", 2: "bigram", 3: "trigram", 4: "4gram"}.get(n)
    if not model:
        raise ValueError("Invalid n value. Only 1, 2, 3, or 4 are allowed.")

    if value_type == "ID":
        column = f"ID_{model}_esidaine"
    elif value_type == "IR":
        column = f"IR_{model}_esidaine"
    else:
        raise ValueError("Invalid value_type. Use 'ID' or 'IR'.")

    summary_df = pd.read_csv('syll_comparison_coupe_esidaine.csv')

    # Check if the column exists, if not, create it with NaN values
    if column not in summary_df.columns:
        summary_df[column] = np.nan


    # If value is a list, update info_rate for each speaker of that language
    if isinstance(value, list):
        # Ensure the length of the value list matches the number of speakers for the language
        language_speakers = summary_df[summary_df['Language'] == language_to_update]
        if len(value) != len(language_speakers):
            raise ValueError("The length of the value list must match the number of speakers for the specified language.")

        for idx, val in zip(language_speakers.index, value):
            summary_df.at[idx, column] = round(val, 3)
    else: 
        summary_df.loc[summary_df['Language'] == language_to_update, column] = value

    # Save the updated DataFrame to the CSV file
    summary_df.to_csv('syll_comparison_coupe_esidaine.csv', sep='\t', index=False)
    print(f"Updated {column}")


def compute_info_rate(info_density, processing_type, language):
    """
    Calculates the information rate based on the provided information density.
    The function assumes:
    - The CSV file is tab-separated (`\t`)
    - There's a column named 'nsyll' representing the number of syllables
    - There's a column named 'phonationtime' representing the phonation time
    Args:
        info_density (float): Information density value
    Returns:
        float: Information rate per second
    """

    counts_df_path = "C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/semantically_similar_texts/ling_units_counts.csv"
    speech_df_path = "C:/Users/emill/Documents/GitHub/Coupe_Expansion/AutomaticSylDetect.csv"

 
    counts_df = pd.read_csv(counts_df_path, sep="\t")
    speech_df = pd.read_csv(speech_df_path, sep="\t")

    speech_df['language'] = speech_df['soundname'].str[:3]
    speech_df['passage_name'] = speech_df['soundname'].str[-2:]

    # Filter the data by the specified language
    speech_df_lang = speech_df[speech_df['language'] == language].copy()
    counts_df_lang = counts_df[counts_df['language'] == language].copy()

    # Merge the two dataframes
    merged_df = pd.merge(
        speech_df_lang,
        counts_df_lang,
        left_on=['language', 'passage_name'],
        right_on=['language', 'passage_name'],
        how='left'
    )

    # Initialize an empty list to store info_rate values
    info_rate_values = []

    # Iterate through each speaker's data
    for _, row in merged_df.iterrows():
        if processing_type == 'sylls': 
            n_units = row['n_syllables']
        elif processing_type == 'phonemes':
            n_units = row['n_phonemes']
        else: 
            warnings.warn("Unknown processing_type. Use 'sylls' or 'phonemes'.")
            return []

        phonationtime = row['phonationtime']

        # Calculate speech rate
        speech_rate = n_units / phonationtime

        # Calculate information rate
        info_rate = info_density * speech_rate
        info_rate_values.append(info_rate)

    return info_rate_values


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


def get_ipa_espeak(word, espeak_code):
    """Get IPA transcription from espeak-ng."""

    # normalize the word
    word = unicodedata.normalize("NFC", word)
    # remove punctuation
    no_punct = word.strip(string.punctuation)

    try:
        result = subprocess.run(
            ['espeak-ng', '-v', espeak_code, '--ipa=3', '-q', word],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"espeak-ng failed on word '{word}': {e}")
        return ""


def phoneme_tokenization(word, language): 
    # Unicode grapheme cluster matcher for phonemes tokenization
    grapheme_pattern = re.compile(r'\X', re.UNICODE)

    espeak_code = load_config(language, "IPA Code")

    # Convert to IPA 
    word_ipa = get_ipa_espeak(word, espeak_code)
    # Clean IPA 
    cleaned_ipa = clean_ipa(word_ipa, True, '', language)
    # print(f"Word: {word}, uncleaned: {word_ipa}, cleaned: {cleaned_ipa}")

    if not cleaned_ipa:
        return None  # remove empty strings

    # Tokenize into phonemes
    phonemes = [match.group() for match in grapheme_pattern.finditer(cleaned_ipa) if match.group() not in (' ', '')]

    return phonemes



def get_word_data(path, processing_type="sylls", clean=True):
    """
    Reads a CSV/TSV/Excel file of phonetically transcribed words and frequencies.
    Splits the words into tokens (syllables, phonemes, or characters), repeated according to frequency.

    Args:
        path (str): Path to the data file.
        processing_type (str): One of 'sylls', 'phonemes', 'chars'.
        clean (bool): Whether to clean IPA before tokenizing.

    Returns:
        list[list[str]]: Tokenized words, repeated by frequency.
    """

    # Extract language from filename robustly
    language = Path(path).parent.name.upper()
    print(f"Language: {language}")

    # Configuration for each language
    config = {
        "FRA": {
            "columns": ["syll", "freqfilms2"],
            "excel_col_filter": "freqfilms2",
            "delimiters": {"sylls": r"[.-]", "phonemes": "-"}
        },
        "DEU": {
            "columns": ["PhonStrsDISC", "Word Mann"],
            "excel_col_filter": "Word Mann",
            "delimiters": {"sylls": "-", "phonemes": "-"}
        },
        "ENG": {"delimiters": {"sylls": r"[-.]", "phonemes": "-"}},
        "CMN": {"delimiters": {"sylls": "_", "phonemes": "-"}},
        "VIE": {"delimiters": {"sylls": "_", "phonemes": "-"}},
        "JPN": {"delimiters": {"sylls": "_", "phonemes": "-"}},
        "YUE": {"delimiters": {"sylls": "_", "phonemes": "-"}},
    }

    if language not in config:
        raise ValueError(f"Unsupported language: {language}")

    lang_cfg = config[language]
    delimiter = lang_cfg.get("delimiters", {}).get(processing_type, "_")

    words = []

    if language in ["FRA", "DEU"]:
        # Load appropriate file format
        if path.endswith(".xlsx"):
            df = pd.read_excel(path, engine="openpyxl")
            df = df[df[lang_cfg["excel_col_filter"]] > 0]
        else:
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
        for _, row in df.iterrows():
            raw_word = str(row[lang_cfg["columns"][0]])
            freq = int(row[lang_cfg["columns"][1]])
            tokens = tokenize(raw_word, processing_type, delimiter, clean, language)

            words.extend([tokens] * freq)

    else:  # Text-based format (e.g. ENG, CMN, VIE, JPN, YUE)
        with open(path, 'r', encoding="utf-8") as file:
            for line in file:
                try:
                    raw_word, freq = line.strip().split('\t')
                    freq = int(freq)
                    tokens = tokenize(raw_word, processing_type, delimiter, clean, language)
                    words.extend([tokens] * int(float(freq)))
                except ValueError:
                    continue  # skip malformed lines

    return words


# --- Mandarin ---
def cmn_to_ipa(text):
    words = jieba.lcut(text)
    ipa_words_list = []
    for word in words:
        pinyins = pypinyin_fn(word, style=Style.TONE3, heteronym=False)
        syllables = [syll[0] for syll in pinyins]
        ipa_words_list.append("_".join(syllables))
    return ipa_words_list


# --- Cantonese ---

def yue_to_ipa(text):
    jyutping_list = pycantonese.characters_to_jyutping(text)
    ipa_words_dict = [jp for jp in jyutping_list if jp]
    ipa_words_list = [word[1] for word in ipa_words_dict if word is not None and word[1] is not None]
    return ipa_words_list

def text_to_ipa(language):
    language_code_dict = {
        'cat': 'ca', 'cmn': 'zh', 'deu': 'de', 'eng': 'en', 'eus': 'eu',
        'fin': 'fi', 'fra': 'fr', 'hun': 'hu', 'ita': 'it', 'jpn': 'ja',
        'kor': 'ko', 'spa': 'es', 'srp': 'sr', 'tha': 'th', 'tur': 'tr',
        'vie': 'vi', 'yue': 'zh-yue'
    }

    # Load configuration CSV
    config_df = pd.read_json("C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/language_config.json")
    config_df.set_index("Language", inplace=True) 

    try:
        lang_cfg = config_df.loc[language]
        espeak_lang = lang_cfg["IPA Code"]
    except KeyError:
        print(f"Language '{language}' not found in configuration.")
        return

    lang = language_key.lower()

    with open(path, "rb") as f:
        text = pickle.load(f)

    for sentence in text:
        if lang == "vie":
            return np.nan

        elif lang == "jpn":
            return np.nan

        elif lang == "tha":
            return np.nan

        elif lang == "cmn":
            print(text)
            print(f"IPA for {lang}: {cmn_to_ipa(text)}")
            return cmn_to_ipa(text)

        elif lang == "yue":
            print(text)
            print(f"IPA for {lang}: {yue_to_ipa(text)}")
            return yue_to_ipa(text)

        elif lang == "kor":
            return np.nan

        else:
            result = subprocess.run(
                ['espeak', '-q', '--ipa3', '-v', espeak_lang, text],
                capture_output=True,
                text=True
            )
            ipa_text = result.stdout.strip().replace('\n', ' ')
            print(ipa_text)
            print(f"IPA for {lang}: {ipa}")
            return ipa.split()

    # --- Apply to CSV ---
    df = pd.read_csv('semantically_similar_texts/semantically_similar_texts_with_ipa.csv')
    df['ipa'] = df.apply(lambda row: text_to_ipa(row['text'], row['language']), axis=1)
    df.to_csv('semantically_similar_texts/semantically_similar_texts_with_ipa.csv', sep='\t', index=False)



    

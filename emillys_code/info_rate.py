from tqdm import tqdm
import pandas as pd
import warnings
import os
from joblib import Parallel, delayed
from functools import partial
import re
import ast
from typing import List, Tuple

from ipa_conversion import load_charsiu_model, parallelize_ipa_generation, get_largest_ipa_corpus
from syllabification import syllable_tokenization_wrapper, get_onsets_ipa

import logging
logging.basicConfig(level=logging.INFO)

TokenizationResult = Tuple[list[str], list[str]]

def compute_info_rate(info_density: float, unit_type: str, language: str):
    """
    Calculate information rate (bits/s) and speech rate (units/s) for a language.

    Combines an information density value (bits per unit) with phonation times 
    and linguistic unit counts to estimate how much information is transmitted 
    per second.

    Parameters
    ----------
    info_density : float
        Information density value in bits per unit (depends on `unit_type`).
    unit_type : {'sylls', 'phones', 'words'}
    language : str
        ISO-3 code of the language (e.g., 'ENG', 'FRA').

    Returns
    -------
    info_rate_values : list of float
        Information rates per second (bits/s) for each speaker and passage.
    speech_rate_values : list of float
        Speech rates (units/s) for each speaker and passage.

    Notes
    -----
    • Requires:
        - `semantically_similar_texts/ling_units_counts.csv`
        - `../AutomaticSylDetect.csv`
    • If IPA parsing fails for words, the row is skipped.  
    • If CSVs are missing, returns empty lists.

    Example
    -------
    >>> compute_info_rate(5.2, "sylls", "ENG")
    ([78.4, 81.2, ...], [15.1, 15.6, ...])
    """

    counts_df_path = "semantically_similar_texts/ling_units_counts.csv"
    speech_df_path = "../AutomaticSylDetect.csv"

 
    try:
        counts_df = pd.read_csv(counts_df_path, sep="\t")
        speech_df = pd.read_csv(speech_df_path, sep="\t")
    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return []


    # Extract language and passage names
    speech_df['language'] = speech_df['soundname'].str[:3]
    speech_df['passage_name'] = speech_df['soundname'].str[-2:].str.upper()
    counts_df['passage_name'] = counts_df['passage_name'].str.upper()

    # Filter by language
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
    speech_rate_values = []

    # Iterate through each speaker's data
    for _, row in merged_df.iterrows():
        if unit_type == 'sylls': 
            n_units = row['n_syllables']
        elif unit_type == 'phones':
            n_units = row['n_phones']
        elif unit_type == 'words':
            try:
                ipa_list = ast.literal_eval(row["ipa"])
            except (ValueError, SyntaxError) as e:
                logging.warning(f"⚠️ Failed to parse 'ipa' field for {language}: {row['ipa']}")
                continue

            if isinstance(ipa_list, list) and all(isinstance(w, str) for w in ipa_list):
                n_units = len(ipa_list)
            else:
                logging.warning(f"⚠️ Malformed 'ipa' entry for {language}: {ipa_list}")
                continue
        else: 
            warnings.warn("Unknown processing_type. Use 'sylls', 'phones', or 'words'.")
            return []

        phonationtime = row['phonationtime']

        # Calculate speech rate
        speech_rate = n_units / phonationtime
        speech_rate_values.append(speech_rate)

        # Calculate information rate
        info_rate = info_density * speech_rate
        info_rate_values.append(info_rate)

    return info_rate_values, speech_rate_values



def count_ling_units(language: str, config_dict: dict, folder: str) -> None:
    """
    Count phones and syllables for semantically similar texts.

    Reads texts, converts them to IPA, applies 
    language-specific syllabification, and saves the results.

    Parameters
    ----------
    language : str
        ISO-3 code of the language (e.g., 'ENG', 'FRA').
    config_dict : dict
        Configuration dictionary. Must include:
          - "Corpus Size" : int, used to select the largest IPA corpus.
    folder : str
        Path to the folder where corpus resources are located.

    Returns
    -------
    None
        Saves results to `semantically_similar_texts/ling_units_counts.csv`.

    Output File
    -----------
    Adds these columns to the file:
      - 'ipa'          : IPA tokens for each text (list)
      - 'n_phones'     : number of phones
      - 'n_syllables'  : number of syllables

    Notes
    -----
    • If the counts file exists, new rows are merged and duplicates dropped.  
    • Errors in missing files raise exceptions.
    • Writes results to ling_units_counts.csv
    """
    input_path = "semantically_similar_texts/semantically_similar_texts.csv"
    output_path = "semantically_similar_texts/ling_units_counts.csv"

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError("Missing required data with semantically similar texts")
        
        tqdm.write("📥 Loading semantically similar texts ...")
        df = pd.read_csv(input_path)
    except FileNotFoundError as e:
        logging.error(e)
        raise

    tokenizer, model = load_charsiu_model()

    largest_corpus_size = config_dict['Corpus Size']
    _, _, ipa_corpus_path = get_largest_ipa_corpus(language, largest_corpus_size, folder)

    if ipa_corpus_path is None:
        raise ValueError("ipa_corpus_path in count_ling_units must not be None.")
    onsets = get_onsets_ipa(language, ipa_corpus_path)

    # Filter rows with respect to language
    df_lang  = df[df["language"] == language].copy()

    all_texts = [str(row["text"]).strip().split() for _, row in df_lang.iterrows()]

    # Each sentence is a list of words or symbols
    texts_formatted = split_sentences(all_texts)

    # Flatten the sentences
    flat_texts = [[word for sentence in text for word in sentence] for text in texts_formatted]
    
    # Ipa conversion
    parallel_ipa = partial(
        parallelize_ipa_generation,
        language=language,
        tokenizer=tokenizer,
        model=model,
        config_dict=config_dict
    )

    flat_ipa_texts = parallel_ipa(flat_texts)

    # Results
    phonemized_data = []
    syllabized_data = []
    n_phones_column = []
    n_syllables_column = []

    tqdm.write("🔄 Splitting data into phones and syllables ...")

    # Flatten all words from all sentences for batch processing
    text_word_counts = []
    words_flat = []
    
    for text in flat_ipa_texts:
        text_word_counts.append(len(text))
        words_flat.extend(text)

    inputs: List[tuple[str, list[str], str]] = [(word, onsets, language) for word in words_flat]

    tasks = [
        delayed(syllable_tokenization_wrapper)(args)
        for args in tqdm(inputs, desc="Tokenizing", ncols=80)
    ]
    
    out = Parallel(n_jobs=15, batch_size=64)(tasks)

    # Runtime validation + normalization to satisfy both Pylance and robustness
    results: List[TokenizationResult] = []

    for r in out:
        if r is None or not isinstance(r, tuple) or len(r) != 2:
            raise TypeError("syllable_tokenization_wrapper must return (phones: list[str], sylls: list[str])")
        phones, sylls = r
        results.append((list(phones), list(sylls)))

    # Reconstruct sentence structure
    idx = 0
    for sent, length in zip(flat_ipa_texts, text_word_counts):
        sentence_results = results[idx:idx + length]
        phones = [p for r in sentence_results for p in r[0]]
        sylls = [s for r in sentence_results for s in r[1]]
        idx += length

        phonemized_data.append(phones)
        syllabized_data.append(sylls)
        n_phones_column.append(len(phones))
        n_syllables_column.append(len(sylls))
    
    # print(f"phones: {phonemized_data}")
    # print(f"syllables: {syllabized_data}")

    # Assign results to DataFrame
    df_lang["ipa"] = flat_ipa_texts
    df_lang["n_phones"] = n_phones_column
    df_lang["n_syllables"] = n_syllables_column

    # Load or create output CSV
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, sep="\t")
        combined_df = pd.concat([existing_df, df_lang], ignore_index=True)
        combined_df.drop_duplicates(subset=["language", "text"], keep="last", inplace=True)
    else:
        combined_df = df_lang

    combined_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    logging.info(f"✅ Linguistic units counted and saved to: {output_path}")


def split_sentences(texts: list[list[str]]) -> list[list[list[str]]]:
    """
    Split tokenized texts into sentences based on punctuation.

    Parameters
    ----------
    texts : list of list of str
        Each text is a list of word tokens. Tokens may contain punctuation.

    Returns
    -------
    output : list of list of list of str
        For each text, a list of sentences, where each sentence is a list 
        of tokens. Sentence boundaries are defined by '.', '!', or '?'.

    Notes
    -----
    • Splits tokens using regex into words/clitics and punctuation.  
    • Keeps commas and punctuation as separate tokens.  

    Example
    -------
    >>> split_sentences([["Hello,", "world!", "How's", "it", "going?"]])
    [[["Hello", ",", "world", "!"],
      ["How's", "it", "going", "?"]]]
    """
    output = []
    for tokens in texts:
        sentence = []
        grouped = []
        for token in tokens:
            # Extract words and punctuation marks separately
            parts = re.findall(r"\w+(?:'\w+)?|[.,!?]", token)
            sentence.extend(parts)
            if any(p in ".!?" for p in parts):
                grouped.append(sentence)
                sentence = []
        if sentence:
            grouped.append(sentence)
        output.append(grouped)
    return output

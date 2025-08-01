from tqdm import tqdm
import pandas as pd
import warnings
from ipa_conversion import load_charsiu_model, parallelize_ipa_generation, get_largest_ipa_corpus
from syllabification import syllable_tokenization_wrapper, get_onsets_ipa
from tqdm import tqdm
import logging
import os
from joblib import Parallel, delayed
from functools import partial
import re
from collections import defaultdict
import ast

logging.basicConfig(level=logging.INFO)


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
        if processing_type == 'sylls': 
            n_units = row['n_syllables']
        elif processing_type == 'phones':
            n_units = row['n_phones']
        elif processing_type == 'words':
            ipa_list = ast.literal_eval(row["ipa"])  
            n_units = len(ipa_list)
        else: 
            warnings.warn("Unknown processing_type. Use 'sylls' or 'phones'.")
            return []

        phonationtime = row['phonationtime']

        # Calculate speech rate
        speech_rate = n_units / phonationtime
        speech_rate_values.append(speech_rate)

        # Calculate information rate
        info_rate = info_density * speech_rate
        info_rate_values.append(info_rate)

    return info_rate_values, speech_rate_values



def count_ling_units(language, config_dict, folder):
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
    onsets = get_onsets_ipa(language, ipa_corpus_path)

    # Filter rows with respect to language
    df_lang  = df[df["language"] == language].copy()

    all_texts = [str(row["text"]).strip().split() for _, row in df_lang.iterrows()]
    # Each sentence is a list of words or symbols
    texts_formatted = split_sentences(all_texts)
    print(f"texts_formatted: {texts_formatted}")

    # Flatten the sentences
    flat_texts = [[word for sentence in text for word in sentence] for text in texts_formatted]
    print(f"flat_texts: {flat_texts}")
    
    # Ipa conversion
    all_texts_ipa = []
    parallel_ipa = partial(
        parallelize_ipa_generation,
        language=language,
        tokenizer=tokenizer,
        model=model,
        config_dict=config_dict
    )

    flat_ipa_texts = parallel_ipa(flat_texts)
    print(f"flat_ipa_texts: {flat_ipa_texts}")

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

    inputs = [(word, onsets, language) for word in words_flat]
    results = Parallel(n_jobs=15, batch_size=64)(
        delayed(syllable_tokenization_wrapper)(args) for args in tqdm(inputs, desc="Tokenizing", ncols=80)
    )

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
    
    print(f"phones: {phonemized_data}")
    print(f"syllables: {syllabized_data}")

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


def split_sentences(texts):
    """
    Splits each list of word tokens in `texts` into sentences.
    Each sentence ends with '.', '?', or '!', which are split off as separate tokens.
    
    Args:
        texts: List of texts, each a list of word tokens (str).
        
    Returns:
        List of texts, each a list of sentences, where each sentence is a list of tokens.
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

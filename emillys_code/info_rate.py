from tqdm import tqdm
import pandas as pd
import warnings
from process_ipa import load_charsiu_model, parallelize_ipa_generation
from syllabification import syllable_tokenization_wrapper, get_onsets_ipa
from tqdm import tqdm
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import word_tokenize


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

    # Iterate through each speaker's data
    for _, row in merged_df.iterrows():
        if processing_type == 'sylls': 
            n_units = row['n_syllables']
        elif processing_type == 'phones':
            n_units = row['n_phones']
        else: 
            warnings.warn("Unknown processing_type. Use 'sylls' or 'phones'.")
            return []

        phonationtime = row['phonationtime']

        # Calculate speech rate
        speech_rate = n_units / phonationtime

        # Calculate information rate
        info_rate = info_density * speech_rate
        info_rate_values.append(info_rate)

    return info_rate_values



def count_ling_units(language):
    input_path = "C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/semantically_similar_texts/semantically_similar_texts.csv"
    output_path = "C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/semantically_similar_texts/ling_units_counts.csv"

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError("Missing required data with semantically similar texts")
        
        tqdm.write("📥 Loading semantically similar texts ...")
        df = pd.read_csv(input_path)
    except FileNotFoundError as e:
        logging.error(e)
        raise

    tokenizer, model = load_charsiu_model()
    onsets = get_onsets_ipa(language)

    # Filter rows with respect to language
    df_lang  = df[df["language"] == language].copy()

    all_texts = [str(row["text"]).strip().split() for _, row in df_lang.iterrows()]

    # Results
    all_ipa = []
    phonemized_data = []
    syllabized_data = []
    n_phones_column = []
    n_syllables_column = []

    ipa_sentences = parallelize_ipa_generation(all_texts, language, tokenizer, model)

    for ipa_sentence in tqdm(ipa_sentences, desc="🔄 Splitting data to phones and syllables", ncols=80):
        inputs = [(ipa, onsets) for ipa in ipa_sentence]

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(syllable_tokenization_wrapper, inputs))

        phones = [p for r in results for p in r[0]]
        sylls = [s for r in results for s in r[1]]

        phonemized_data.append(phones)
        syllabized_data.append(sylls)
        all_ipa.append(ipa_sentence)
        n_phones_column.append(len(phones))
        n_syllables_column.append(len(sylls))

    # Assign results to DataFrame
    df_lang["ipa"] = all_ipa
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
    print(f"✅ Linguistic units counted and saved to: {output_path}")
from math import log2
import pandas as pd
import warnings
import os
from pathlib import Path
import numpy as np
import pandas as pd
from syllabification import phone_tokenization, syllable_tokenization, parse_to_phones_and_sylls
from process_ipa import generate_ipa, load_charsiu_model


def update_values_in_csv(language_to_update, value, n, value_type, text_type):
    model = {1: "unigram", 2: "bigram", 3: "trigram", 4: "4gram"}.get(n)
    if not model:
        raise ValueError("Invalid n value. Only 1, 2, 3, or 4 are allowed.")

    if value_type == "ID" or value_type == "IR":
        ling_unit_comparison_column = f"{value_type}_{model}"
        inter_intra_comparison_column = f"{text_type}_{value_type}_{model}"
    else:
        raise ValueError("Invalid value_type. Use 'ID' or 'IR'.")


    ling_unit_comparison_df = pd.read_csv('C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/produced_data/ling_unit_comparison.csv')
    inter_intra_comparison_df = pd.read_csv('C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/produced_data/word_sent_comparison.csv')

    # Check if the column exists, if not, create it with NaN values
    if ling_unit_comparison_column not in ling_unit_comparison_df.columns:
        ling_unit_comparison_df[ling_unit_comparison_column] = np.nan
    if inter_intra_comparison_column not in inter_intra_comparison_df.columns:
        inter_intra_comparison_df[inter_intra_comparison_column] = np.nan

    print(inter_intra_comparison_df.columns)
    print(ling_unit_comparison_df.columns)

    # If value is a list, update info_rate for each speaker of that language
    if isinstance(value, list):
        # Ensure the length of the value list matches the number of speakers for the language
        language_speakers = ling_unit_comparison_df[ling_unit_comparison_df['Language'] == language_to_update]
        if len(value) != len(language_speakers):
            raise ValueError("The length of the value list must match the number of speakers for the specified language.")

        for idx, val in zip(language_speakers.index, value):
            ling_unit_comparison_df.at[idx, ling_unit_comparison_column] = round(val, 3)
            inter_intra_comparison_df.at[idx, inter_intra_comparison_column] = round(val, 3)
    else: 
        ling_unit_comparison_df.loc[ling_unit_comparison_df['Language'] == language_to_update, ling_unit_comparison_column] = value
        inter_intra_comparison_df.loc[inter_intra_comparison_df['Language'] == language_to_update, inter_intra_comparison_column] = value

    # Save the updated DataFrame to the CSV file
    ling_unit_comparison_df.to_csv('C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/produced_data/ling_unit_comparison.csv', index=False)
    inter_intra_comparison_df.to_csv('C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/produced_data/inter_intra_comparison.csv', index=False)

    print(f"Updated {ling_unit_comparison_column}")
    print(f"Updated {inter_intra_comparison_column}")


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


def check_data_avilability(language, processing_type): 

    """
    Checks whether required processed data and unit count data are available for a given language.

    Args:
        processing_type (str): One of 'phones' or 'sylls'.
        text_type (str): Currently unused, but reserved for future use.
        language (str): Language code (e.g., 'FRA').
        folder (str): Directory containing processed JSON files.

    Returns:
        str: Path to the prepared data file.

    Raises:
        ValueError: If input type is invalid or required files/data are missing.
        KeyError: If the raw corpus is not registered.
    """
    folder = f"produced_data/{language}"

    if processing_type not in ['phones', 'sylls']:
        raise ValueError("❌ Invalid processing type. Use 'phones' or 'sylls'.")

    filename = f"phonemized_{language}.json" if processing_type == 'phones' else f"syllabized_{language}.json"
    input_path = os.path.join(folder, filename)

    # Check if the input file exists

    if not os.path.exists(input_path):
        print(f"❌ No prepared {processing_type} data found for {language} at {input_path}.")
        try: 
            data_path = load_config(language, 'Sentence Data')
            print(f"✅ Raw sentence-level corpus found at {data_path}.")
            print(f"👉 Please run `parse_to_phones_and_sylls('{language}')` to generate the required data.")
            response = input(f"❓ Do you want to run `parse_to_phones_and_sylls('{language}')` [y/n]: ").strip().lower()
            if response in ['y', 'yes']:
                parse_to_phones_and_sylls(language)        
        except KeyError:
            raise KeyError(f"❌ No raw sentence-level corpus registered for '{language}'. Please choose a different language.")
        return None


    ling_unit_count_csv_path = (
        "C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/semantically_similar_texts/ling_units_counts.csv"
    )

    check_failed = False

    # Check if file exists 
    if not os.path.isfile(ling_unit_count_csv_path):
        print(f"❌ Linguistic unit counts CSV not found at: {ling_unit_count_csv_path}. "
              f"👉 Please run `count_ling_units('{language}')` to generate it."
        )
        check_failed = True
        
     # Load and check content
    df = pd.read_csv(ling_unit_count_csv_path, sep=None, engine='python')  # auto-detect delimiter
    
    if language not in df["language"].unique() or "n_phones" not in df.columns or "n_syllables" not in df.columns:
        print(f"❌ Linguistic unit counts for '{language}' not found in the respective csv file. "
              f" 👉 Please run count_ling_units ('{language}') first"
        )
        check_failed = True

    # Check for non-null values in 'n_phones' and 'n_syllables'
    subset = df[df["language"] == language]
    if subset["n_phones"].isnull().any() or subset["n_syllables"].isnull().any():
        print(f"❌ Linguistic unit counts for '{language}' not found in the respective csv file. "
              f" 👉 Please run count_ling_units{language} first"
        )
        check_failed = True
    
    if check_failed:
        response = input(f"❓ Do you want to run `count_ling_units('{language}')` [y/n]: ").strip().lower()
        if response in ['y', 'yes']:
            count_ling_units(language)
        else: raise FileNotFoundError(f" Stopping execution.")

    return input_path



def count_ling_units(language):
    print(f"Running linguistic unit counting for {language}. This may take a while...")

    # Load CSV and language config
    df = pd.read_csv("C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/semantically_similar_texts/semantically_similar_texts.csv")

    tokenizer, model = load_charsiu_model()

    # Filter rows with respect to language
    df = df[df["language"] == language].copy()  

    # Result lists
    ipa_column = []
    n_phones_column = []
    n_syllables_column = []


    for idx, row in df.iterrows():
        text = str(row["text"]).strip()

        # IPA transcription and phoneme an syllable count
        all_ipa = []
        all_phones = []
        all_syllables = []

        for word in text.split():

            # Convert to ipa 
            cleaned_ipa, _ = generate_ipa(word, language, tokenizer, model)

            if not cleaned_ipa:
                continue # removal of empty strings
            all_ipa.append(cleaned_ipa)

            # Tokenize into phones
            phones = phone_tokenization(cleaned_ipa)
            all_phones.extend(phones)

            # Tokenize into syllables
            syllables = syllable_tokenization(cleaned_ipa, language)
            all_syllables.extend(syllables)

        ipa_column.append(all_ipa)
        n_phones_column.append(len(all_phones))
        n_syllables_column.append(len(all_syllables))

    # Add to dataframe
    df["ipa"] = ipa_column
    df["n_phonemes"] = n_phones_column
    df["n_syllables"] = n_syllables_column

    # Save result
    output_path = "semantically_similar_texts/ling_units_counts.csv"
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"✅ Linguistic units counted and saved to: {output_path}")







    

import pandas as pd
from pathlib import Path
import numpy as np
from syllabification import parse_to_phones_and_sylls
import sys
from info_rate import count_ling_units
import os
import logging
from functools import partial
import re
from itertools import product
from glob import glob
from IPython.display import display

logging.basicConfig(level=logging.INFO)

def load_config(language):
    config_path = Path("language_config.json")
    config_df = pd.read_json(config_path)
    config_df.set_index("Language", inplace=True)
    config_dict = {}

    try:
        lang_cfg = config_df.loc[language]
        config_dict = lang_cfg.to_dict()
        return config_dict
    except KeyError:
        print(f"Language '{language}' is not supported in language_config.json.")
        return None

def validate_structure(data, text_type, processing_type, context="root", nesting=None):
    # Set expected nesting 
    if nesting is None:
        if processing_type in ("sylls", "phones"):
            nesting = [list, list, list, str]  # sentences → words → units (strings)
        elif processing_type == "words":
            nesting = [list, list, str]        # sentences → words (strings)
        else:
            raise ValueError(f"Invalid processing type: {processing_type}")

    expected_type = nesting[0]

    # Check type at this level
    if not isinstance(data, expected_type):
        raise TypeError(f"{context} expected {expected_type.__name__}, got {type(data).__name__}")

    # If more levels remain, recurse
    if len(nesting) > 1:
        for i, elem in enumerate(data):
            validate_structure(elem, text_type, processing_type, f"{context}[{i}]", nesting[1:])




def update_values_in_csv(language_to_update, value, n, value_type, text_type, processing_type,
                         round_digits=3):
    # Validate model type
    model = {1: "unigram", 2: "bigram", 3: "trigram", 4: "quadgram"}.get(n)
    if not model:
        raise ValueError("Invalid n value. Only 1, 2, 3, or 4 are allowed.")

    # Validate value_type
    allowed_types = {"ID", "IR", "SR"}
    if value_type not in allowed_types:
        raise ValueError(f"Invalid value_type. Use one of {allowed_types}.")

    # Construct column names
    ling_unit_column = f"{processing_type}_{value_type}_{model}_esidaine"
    inter_intra_column = f"{text_type}_{processing_type}_{value_type}_{model}"

    # File paths
    base_path ='produced_data_large_corpus'
    ling_file = os.path.join(base_path, 'ling_unit_comparison.csv')
    inter_intra_file = os.path.join(base_path, 'inter_intra_comparison.csv')

    # Load CSVs
    unit_df = pd.read_csv(ling_file)
    inter_intra_df = pd.read_csv(inter_intra_file)

    # Ensure columns exist
    for df, col in [(unit_df, ling_unit_column), (inter_intra_df, inter_intra_column)]:
        if col not in df.columns:
            df[col] = np.nan


    # Take values from Coupé et. al data
    original_df = pd.read_csv("../InfoRateData.csv", sep="\t")

    unit_languages = unit_df['Language'].to_list()
    inter_intra_languages = inter_intra_df['Language'].to_list()

    # Create a list of (name, DataFrame object) pairs to update
    dfs_to_update = []
    if language_to_update not in unit_languages:
        dfs_to_update.append(("unit_df", unit_df))
    if language_to_update not in inter_intra_languages:
        dfs_to_update.append(("inter_intra_df", inter_intra_df))

    # Filter speaker + passage data from Coupé et al.
    filtered_speaker_data = original_df[original_df['Language'] == language_to_update]
    speaker_passages = filtered_speaker_data[['Speaker', 'Text']].drop_duplicates()

    # Apply to each relevant DataFrame
    for name, df in dfs_to_update:
        new_rows = []

        for _, row_data in speaker_passages.iterrows():
            speaker_id = row_data['Speaker']
            passage = row_data['Text']

            row = {
                'Speaker': speaker_id,
                'Language': language_to_update,
                'Text': passage
            }

            for col in df.columns:
                if col not in row:
                    row[col] = np.nan

            new_rows.append(row)

        updated_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # Assign back to the appropriate variable
        if name == "unit_df":
            unit_df = updated_df
        elif name == "inter_intra_df":
            inter_intra_df = updated_df

    # Get relevant rows
    unit_mask = unit_df['Language'] == language_to_update
    inter_mask = inter_intra_df['Language'] == language_to_update

    # Ensure matching number of rows (per speaker)
    ling_indices = unit_df[unit_mask].index
    inter_indices = inter_intra_df[inter_mask].index

    if isinstance(value, list):
        if len(value) != len(ling_indices):
            raise ValueError("Length of value list must match number of speakers for the language.")

        for idx, val in zip(ling_indices, value):
            unit_df.at[idx, ling_unit_column] = round(val, round_digits)
        for idx, val in zip(inter_indices, value):
            inter_intra_df.at[idx, inter_intra_column] = round(val, round_digits)

    else:
        val = round(value, round_digits)
        unit_df.loc[unit_mask, ling_unit_column] = val
        inter_intra_df.loc[inter_mask, inter_intra_column] = val

    # Save the updated dataframes
    unit_df.to_csv(ling_file, index=False)
    inter_intra_df.to_csv(inter_intra_file, index=False)



def ask_question(question, function_to_run, language):
    """
    Asks a question and runs a function based on the user's response.
    Args:
        question (str): The question to ask the user.
        function_to_run (function): The function to run if the user answers 'yes'.
        language (str): The language code to pass to the function.
    """
    print(f"{question}")
    answer = input(f"{question}").strip().lower()
    print(f"> {answer}")
    if answer in ['y', 'yes']:
        function_to_run(language)
        return True
    else:
        print(f"📂 No data available. Skipping...")
        return False



def check_data_availability(language, processing_type, config_dict): 
    """
    Checks whether required processed data and unit count data are available for a given language.

    Args:
        language (str): Language code (e.g., 'FRA').
        processing_type (str): One of'phones' or 'sylls'.

    Returns:
        Path or None: Path to the prepared data file, or None if checks failed.

    Raises:
        ValueError: If processing_type is invalid.
        KeyError: If no raw corpus is registered for the language.
    """

    if processing_type not in ['phones', 'sylls', 'words']:
        raise ValueError("❌ Invalid processing type. Use 'phones', 'words' or 'sylls'.")

    folder = Path("produced_data") / language / processing_type
    matches = list(folder.glob(f"{'phonized' if processing_type == 'phones' else 'syllabified'}_{language}_size:*.pkl"))

    if not matches:
        print(f"❌ No prepared {processing_type} data found for {language} at {folder}.")
        sys.stdout.flush()
        if not ask_question(
            f"❓ Run parse_to_phones_and_sylls('{language}') now to generate the required data? [y/n]: ",
            partial(parse_to_phones_and_sylls, config_dict=config_dict, folder=f"produced_data_large_corpus/{language}"),
            language
        ):
            return None  
        matches = list(folder.glob(f"{'phonized' if processing_type == 'phones' else 'syllabified'}_{language}_size:*.pkl"))
        if not matches:
            print(f"⚠️ Still no data found for {processing_type} after generation.")
            return None         

    input_path = matches[0]
    ling_unit_count_csv_path = Path("semantically_similar_texts/ling_units_counts.csv")
    check_failed = False

    if not ling_unit_count_csv_path.is_file():
        print(f"❌ Linguistic unit counts CSV not found at: {ling_unit_count_csv_path}.")
        sys.stdout.flush()
        check_failed = True
    else:  
        df = pd.read_csv(ling_unit_count_csv_path, sep=None, engine='python')
        unit_col = 'n_phones' if processing_type == 'phones' else 'n_syllables'
        required_columns = {"language", unit_col}

        if not required_columns.issubset(df.columns) or language not in df["language"].unique():
            print(f"❌ Required columns or language entry missing in data file.")
            sys.stdout.flush()
            check_failed = True
        else:
            subset = df[df["language"] == language]
            if subset[unit_col].isnull().any():
                print(f"❌ Missing values for '{language}' in required data file.")
                sys.stdout.flush()
                check_failed = True

    if check_failed:
        if not ask_question(
            f"❓ Run count_ling_units('{language}') now to generate the required data? [y/n]: ",
            partial(count_ling_units, config_dict=config_dict, folder=f"produced_data_large_corpus/{language}"),
            language
        ):
            return None
        elif ling_unit_count_csv_path.is_file():
            check_failed = False

    # Final recheck
    if input_path.exists() and not check_failed:
        print(f"✅ All checks passed. Passing {input_path}")
        return input_path
    elif input_path.exists() and check_failed:
        print(f"⚠️ Data file exists but unit count validation may have failed.")
        return None
    else:
        logging.error(f"❌ Data generation failed")
        return None
    

def check_expected_values(df): 
    # Define expected columns 
    expected_processing_types = ['phones', 'sylls', 'words']
    text_types = ['within_words', 'across_sentences']
    n_values = ['unigram', 'bigram', 'trigram', 'quadgram']
    metrics = ['ID', 'IR', 'SR']

    # Generate expected column names
    expected_columns = {
        f"{tt}_{pt}_{metric}_{n}"
        for tt, pt, n, metric in product(text_types, expected_processing_types, n_values, metrics)
        if not (tt == 'within_words' and pt == 'words')  # Exclude invalid combo
    }

    # Find actual columns (ignoring metadata)
    actual_columns = set(df.columns) - {'Speaker', 'Language', 'Text'}

    # Compare
    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns

    print(f"✅ Total expected columns: {len(expected_columns)}")
    print(f"✅ Total actual columns: {len(actual_columns)}")

    if missing:
        print("\n❌ Missing columns:")
        for col in sorted(missing):
            print(f"  - {col}")
    else:
        print("\n✅ All expected combinations are present.")

    if extra:
        print("\n⚠️ Extra unexpected columns:")
        for col in sorted(extra):
            print(f"  - {col}")

    # Check for entirely empty columns per language
    result = {}
    feature_cols = [col for col in df.columns if col not in ['Speaker', 'Language', 'Text']]
    
    for lang, group in df.groupby('Language'):
        empty_cols = [
            col for col in feature_cols
            if group[col].apply(lambda x: pd.isna(x) or str(x).strip().lower() in {"", "nan", "none"}).all()
        ]

        if empty_cols:
            result[lang] = empty_cols

    if result:
        print("\n⚠️ Columns present but entirely empty (all NaN), per language:")
        for lang, cols in sorted(result.items()):
            print(f"  {lang}:")
            for col in sorted(cols):
                print(f"    - {col}")
    else:
        print("\n✅ No expected columns are entirely empty for any language.")
    

from pathlib import Path
import os

def clean_corpus_size_files(base_folder, language_codes, corpus_size, processing_types):
    """
    Deletes files matching 'size:{corpus_size}' for each language:
    - In base_folder / language
    - In base_folder / language / processing_type

    Args:
        base_folder (str or Path): Base directory (e.g., 'produced_data')
        language_codes (list of str): List of language ISO codes (e.g., ['DEU', 'ENG'])
        corpus_size (int or str): Corpus size string to match in filenames (e.g., 100, 500)
        processing_types (list of str): List of processing type folders (e.g., ['phones', 'sylls'])
    """
    base_folder = Path(base_folder)
    pattern = f"size:{corpus_size}"
    files_deleted = 0

    for language in language_codes:
        lang_folder = base_folder / language

        # First: clean files directly in base_folder / language
        if lang_folder.exists():
            for file in lang_folder.iterdir():
                if file.is_file() and pattern in file.name:
                    try:
                        file.unlink()
                        logging.info(f"🗑️ Deleted {file}")
                        files_deleted += 1
                    except Exception as e:
                        logging.warning(f"❌ Could not delete {file.name}: {e}")
        else:
            logging.warning(f"⚠️ Folder not found: {lang_folder}")

        # Second: clean files in each processing_type subfolder
        for processing_type in processing_types:
            target_folder = lang_folder / processing_type
            if target_folder.exists():
                for file in target_folder.iterdir():
                    if file.is_file() and pattern in file.name:
                        try:
                            file.unlink()
                            logging.info(f"🗑️ Deleted {file}")
                            files_deleted += 1
                        except Exception as e:
                            logging.warning(f"❌ Could not delete {file.name}: {e}")
            else:
                logging.warning(f"⚠️ Folder not found: {target_folder}")

    if files_deleted == 0:
        logging.warning(f"ℹ️ No files matched pattern 'size:{corpus_size}'")


def create_minimal_summary(df_summary, corpus_size): 

    print(f"\nResults for corpus size {corpus_size}:\n")

    table = df_summary.pivot_table(
        values="Value",
        index=["Language", "UnitType"],
        columns=["TextType", "Metric", "n"],
        aggfunc="mean"
    )

    #  Normalize the color gradient globally across the whole table
    vmin = df_summary["Value"].min()
    vmax = df_summary["Value"].max()

    # Display styled table
    display(
        table.style
        .set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        )
        .set_properties(**{'text-align': 'center'})
        .background_gradient(cmap="Blues", vmin=vmin, vmax=vmax)
        .format("{:.2f}")
    )











    

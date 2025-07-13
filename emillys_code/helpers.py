import pandas as pd
from pathlib import Path
import numpy as np
from syllabification import parse_to_phones_and_sylls
import sys
from info_rate import count_ling_units
import os
import logging
from functools import partial

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


def update_values_in_csv(language_to_update, value, n, value_type, text_type, processing_type,
                         round_digits=3):
    # Validate model type
    model = {1: "unigram", 2: "bigram", 3: "trigram", 4: "4gram"}.get(n)
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
    base_path ='produced_data'
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
    original_df = pd.read_csv("C:/Users/emill/Documents/GitHub/Coupe_Expansion/InfoRateData.csv", sep="\t")

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

    print("Checking data availability...")

    if processing_type not in ['phones', 'sylls']:
        raise ValueError("❌ Invalid processing type. Use 'phones' or 'sylls'.")

    folder = Path("produced_data") / language / processing_type
    filename = f"phonized_{language}.pkl" if processing_type == 'phones' else f"syllabified_{language}.pkl"
    input_path = folder / filename

    # Check if the input file exists
    if not input_path.exists():
        print(f"❌ No prepared {processing_type} data found for {language} at {input_path}.")
        print(f"👉 Please run parse_to_phones_and_sylls('{language}') to generate the required data.")
        sys.stdout.flush() #  Force the print to show up immediately
        if not ask_question(
            f"❓ Run parse_to_phones_and_sylls('{language}') now? [y/n]: ",
            partial(parse_to_phones_and_sylls, config_dict=config_dict),
            language
        ):
            return None               


    ling_unit_count_csv_path = Path("semantically_similar_texts/ling_units_counts.csv"
    )

    check_failed = False

    # Check if file exists 
    if not ling_unit_count_csv_path.is_file():
        print(f"❌ Linguistic unit counts CSV not found at: {ling_unit_count_csv_path}.\n 👉 Please run count_ling_units('{language}') to generate it."
        )
        sys.stdout.flush() #  Force the print to show up immediately
        check_failed = True
    
    else:  
        # Load and check content
        df = pd.read_csv(ling_unit_count_csv_path, sep=None, engine='python')  # auto-detect delimiter
        unit_col = 'n_phones' if processing_type == 'phones' else 'n_syllables'
        required_columns = {"language", unit_col}

        if not required_columns.issubset(df.columns) or language not in df["language"].unique():
            print(f"❌ Required columns or language entry missing in data file.\n"
                  f"👉 Please run count_ling_units('{language}') first.")
            sys.stdout.flush() #  Force the print to show up immediately
            check_failed = True
        else:
            subset = df[df["language"] == language]
            if subset[unit_col].isnull().any():
                print(f"❌ Missing values for '{language}' in required data file.\n"
                      f"👉 Please run count_ling_units('{language}') again.")
                sys.stdout.flush() #  Force the print to show up immediately
                check_failed = True
    
    if check_failed:
        if not ask_question(
            f"❓ Run count_ling_units('{language}') now? [y/n]: ",
            partial(count_ling_units, config=config_dict),
            language
        ):
            return None
        else: 
            if ling_unit_count_csv_path.is_file():
                check_failed = False
    
    # Final recheck after attempted fixes
    if input_path.exists() and not check_failed:
        print(f"✅ All checks passed. Passing {input_path}")
        return input_path
    elif input_path.exists() and check_failed:
        print(f"⚠️ Data file exists but unit count validation may have failed.")
        return None  # OR return None depending on strictness
    else:
        logging.error(f"❌ Data generation failed")
        return None











    

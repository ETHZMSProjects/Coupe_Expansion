import pandas as pd
from pathlib import Path
import numpy as np
from syllabification import parse_to_phones_and_sylls
import sys
from config_loader import load_config
from info_rate import count_ling_units
import os


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
    base_path='C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/produced_data'
    ling_file = os.path.join(base_path, 'ling_unit_comparison.csv')
    inter_intra_file = os.path.join(base_path, 'inter_intra_comparison.csv')

    # Load CSVs
    unit_df = pd.read_csv(ling_file)
    inter_intra_df = pd.read_csv(inter_intra_file)

    # Ensure columns exist
    for df, col in [(unit_df, ling_unit_column), (inter_intra_df, inter_intra_column)]:
        if col not in df.columns:
            df[col] = np.nan

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

    # Save updates
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



def check_data_availability(language, processing_type): 
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

        data_path = load_config(language, 'Sentence Data')
        print(f"👉 Please run parse_to_phones_and_sylls('{language}') to generate the required data.")
        sys.stdout.flush() #  Force the print to show up immediately
        if not ask_question(f"❓ Do you want to run it now? [y/n]:", parse_to_phones_and_sylls, language): 
            return None               


    ling_unit_count_csv_path = Path(
        "C:/Users/emill/Documents/GitHub/Coupe_Expansion/emillys_code/semantically_similar_texts/ling_units_counts.csv"
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
        if not ask_question(f"❓ Do you want to run it now? [y/n]:", count_ling_units, language): 
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
        print(f"❌ Data generation failed")
        return None











    

from math import log2
import pandas as pd
import warnings
import re
from pathlib import Path
import numpy as np

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
    summary_df.to_csv('syll_comparison_coupe_esidaine.csv', index=False)
    print(f"Updated {column}")


def get_info_rate(info_density, language):
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
    file_path = r"C:/Users/emill/Documents/GitHub/Coupe_Expansion/AutomaticSylDetect.csv"
    df = pd.read_csv(file_path, sep="\t")  # Assuming the file is tab-separated


    # Filter rows where the Language matches
    df['Language'] = df['soundname'].str[:3]
    language_df = df[df['Language'] == language]

    # Initialize an empty list to store info_rate values
    info_rate_values = []

    # Iterate through each speaker's data
    for _, row in language_df.iterrows():
        nsyll = row['nsyll']
        phonationtime = row['phonationtime']

        # Calculate speech rate
        speech_rate = nsyll / phonationtime

        # Calculate information rate
        info_rate = info_density * speech_rate
        info_rate_values.append(info_rate)

    return info_rate_values


def get_word_data(path):
    """
    Reads a CSV file containing word frequency data and returns a dictionary 
    mapping each word to its frequency.

    The function assumes:
    - The CSV file is tab-separated (`\t`)
    - There's a column with the syllabified phonetically transcribed words
    - There's a column with the corresponding word frequencies

    Args:
        path (str): Path to the CSV file

    Returns:
        list: A list of lists where each sublist is a word split into syllables,
              repeated according to its frequency
    """
    # Extract language from the file name
    language = path.split("/")[2]
    print(f"Language: {language}")

    words = []

    if language in ["FRA", "DEU"]:
        if language == "FRA":
            columns = ["syll", "freqfilms2"]
            delimiter = r"[.-]"
     
        if language == "DEU":
            columns = ["PhonStrsDISC", "Word Mann"]
            delimiter = r"-"
    
            # Use the correct reader for Excel
        if path.endswith(".xlsx"):
            df = pd.read_excel(path, engine="openpyxl") 
            df = df[df["Word Mann"] > 0]
        else:
            df = pd.read_csv(path, sep="\t", encoding="utf-8")


        for _, row in df.iterrows():
            # Remove unwanted characters
            # Basic list of German IPA vowels for checking
            ipa_vowels = "aeiouyɛœøɐɪʊəɔ"

            # Filter syllables 
            cleaned = re.sub(r"[§.'=/()|&]", "", row[columns[0]])
            cleaned = cleaned.replace("&p", "")  
            row[columns[0]] = cleaned.strip()

            # Split into syllables
            word_sylls = re.split(delimiter, row[columns[0]]) 
            
            # Get the frequency value
            freq = int(row[columns[1]])

            # Replicate the syllables by the frequency and add them to the list
            words.extend([word_sylls] * freq)
    elif language  in ["CMN", "VIE", "JPN", "YUE", "ENG"]:
        with open(path, 'r', encoding="utf-8") as file:
            for line in file:
                # Split the line by tab character
                word, freq = line.strip().split('\t')
                if language == "ENG":
                    delimiter = r'-.'
                    unwanted_chars = r"[§.=_]"
                else: 
                    delimiter = r"[_]"
                    unwanted_chars = r"[§.=]"
                
                # Remove unwanted characters
                cleaned = re.sub(unwanted_chars, "", word)
                word = cleaned.strip()

                # Split into syllables
                word_sylls = re.split(delimiter, word)

                freq = int(float(freq))

                # Replicate the syllables by the frequency and add them to the list
                words.extend([word_sylls] * freq)

    return words
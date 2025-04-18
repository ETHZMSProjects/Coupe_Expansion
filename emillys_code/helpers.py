from math import log2
import pandas as pd
import warnings
import re
from pathlib import Path

def update_values_in_csv(language_to_update, value, n):
    if n == 1:
        model = "unigram"
    elif n == 2:
        model = "bigram"
    elif n == 3:
        model = "trigram"
    elif n == 4:
        model = "4gram"
    else:
        raise ValueError("Invalid n value. Only 1, 2, 3, or 4 are allowed.")

    column = f"ID_{model}_esidaine"

    # Check if the column exists, if not, create it with NaN values
    if column not in summary_df.columns:
        summary_df[column] = pd.NA

    #Read the summary CSV file
    summary_df = pd.read_csv('syll_comparison_coupe_esidaine.csv')

    # Update the respective column (value type) for that language
    summary_df.loc[summary_df['Language'] == language_to_update, column] = value

    # Step 5: Save the updated DataFrame to the CSV file
    summary_df.to_csv('syll_comparison_coupe_esidaine.csv', index=False)

    print(f"Updated {column} for language {language_to_update} with value {value}.")


def get_info_rate(info_density):
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
    file_path = r"C:\Users\emill\Documents\GitHub\Coupe_Expansion\InfoRateData.csv"
    
    # Read the CSV file
    df = pd.read_csv(file_path, sep="\t")  # Assuming the file is tab-separated

    # Extract the columns "nsyll" and "phonationtime"
    nsyll = df['nsyll']
    phonationtime = df['phonationtime']

    speech_rate = nsyll / phonotationtime

    # Calculate information rate per second
    info_rate = info_density * speech_rate 
    return info_rate


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
    language = Path(path).parent.name 
    print(f"Language: {language}")

    words = []

    if language == "FRA":
        columns = ["syll", "freqfilms2"]
        delimiter = r"[.-]"
        # Read the file
        df = pd.read_csv(path, sep="\t", encoding="utf-8") 
        print(df.head())
     
        for _, row in df.iterrows():
            if pd.isna(row[columns[0]]):
                print(f"Skipping row with NaN in column {columns[0]}")  
                continue

            # Split into syllables
            word_sylls = re.split(delimiter, row[columns[0]]) 
            print(f"word_sylls: {word_sylls}")
            
            # Get the frequency value
            freq = int(row[columns[1]])

            # Replicate the syllables by the frequency and add them to the list
            words.extend([word_sylls] * freq)

    elif language  in ["CMN", "VIE", "JPN"]:
        delimiter = r"[_]"
        with open(path, 'r', encoding="utf-8") as file:
            for line in file:
                # Split the line by tab character
                word, freq = line.strip().split('\t')
                freq = int(freq)

                # Split into syllables
                word_sylls = re.split(delimiter, word)

                # Replicate the syllables by the frequency and add them to the list
                words.extend([word_sylls] * freq)

    return words

def vietnamese_ID_for_normalization(): 
    """
    Loads a tab-separated CSV file and extracts all Information Density (ID) values 
    for Vietnamese speakers. Prints each ID and computes the average ID, which can 
    be used as a baseline for normalization.

    The function assumes:
    - The CSV file is tab-separated (`\t`)
    - There's a column named 'Language' with 'VIE' representing Vietnamese
    - There's a column named 'ID' containing the information density values

    Returns:
        float: The average ID value for Vietnamese speakers
    """
    df = pd.read_csv(r"C:\Users\emill\Documents\GitHub\Coupe_Expansion\InfoRateData.csv", sep='\t')

    # Filter rows where the Language is Vietnamese
    vietnamese_df = df[df['Language'] == 'VIE'] 

    #print("All Vietnamese ID values:")
    #print(vietnamese_df['ID'])

    # Compute the average ID for Vietnamese speakers
    ID_vietnamese = vietnamese_df['ID'].mean()

    print("Baseline Vietnamese ID:", ID_vietnamese)

    return ID_vietnamese
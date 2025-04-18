from math import log2
import pandas as pd
import warnings
import re
from pathlib import Path

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
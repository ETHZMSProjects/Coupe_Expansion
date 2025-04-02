import pandas as pd
import spacy

# Load the French language model from spaCy
nlp = spacy.load("fr_core_news_sm")

def tokenize(sentence):
    """
    Tokenizes a given sentence into individual words using spaCy.

    Args:
        sentence (str): The sentence to be tokenized.

    Returns:
        list: A list of lowercase words from the sentence, excluding non-alphabetic tokens.
    """
    # Load French model for tokenization
    nlp = spacy.load("fr_core_news_sm")  
    doc = nlp(sentence)
    
    # Extract and return the words, filtering out non-alphabetic tokens
    tokenized_sentence = [token.text.lower() for token in doc if token.is_alpha]
    return tokenized_sentence

def syllabify_sentences(tokenized_sentence, language="French", preview=True):
    """
    Converts a list of tokenized words into a sequence of transcribed syllables.
    The syllables are retrieved from a lexicon specific to the language.

    Args:
        tokenized_sentence (list): A list of tokenized words from a sentence.
        language (str): The language of the sentence (default is "French").
        preview (bool): If True, prints a preview of the transcribed sentence (default is True).

    Returns:
        list: A list of syllables for each word in the sentence.
    """
    if language == 'French':
        # Path to the Lexique383 dataset for French words and their syllables
        path = "french_lexique/Lexique383.tsv"

        # Read the dataset into a DataFrame
        feature_df = pd.read_csv(path, sep="\t", encoding="utf-8")
        
        # Drop rows with missing orthographic or syllable features
        feature_df = feature_df.dropna(subset=['ortho', 'syll'])

        # Print some rows and columns for inspection (can be removed in production)
        print(feature_df.head(20))  
        print(feature_df.columns)  
        print(feature_df[['ortho', 'syll']].head(40)) 

        # Build a lookup dictionary: {word: syllables}
        word_to_feature_dict = dict(zip(feature_df['ortho'].str.lower(), feature_df['syll']))
    else:
        # Raise an error if the language is not supported
        raise ValueError(f"Language '{language}' not supported yet.")

    # Prepare the list to store the syllabified sentence
    transcribed_sentence = []
    features_per_word = []

    # Iterate through each word in the tokenized sentence
    for word in tokenized_sentence:
        # Get syllables for the word, or return the word itself if not found in the dictionary
        transcribed_word = word_to_feature_dict.get(word, word)  
        
        # If syllables are combined with a dot, split them
        if '.' in transcribed_word:
            splitted_word = transcribed_word.split('.')  
            features_per_word.extend(splitted_word)
        else:
            features_per_word.append(transcribed_word)

    # Append the syllables for the entire sentence
    transcribed_sentence.append(features_per_word)

    # If preview is True, print the transcribed sentence
    if preview:
        print(transcribed_sentence)

    return transcribed_sentence


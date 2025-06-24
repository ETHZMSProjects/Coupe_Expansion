import os
import regex as re
import pickle
from collections import Counter
from process_ipa import generate_ipa
from process_ipa import load_charsiu_model
import json
from process_ipa import load_config
from tqdm import tqdm

# --- Main Processing Function ---
def parse_to_phones_and_sylls(language):
    """
    Parses sentences to phonemes and syllables using external phone_tokenization
    and CharsiuG2P for syllabification.

    Args:
        language (str): The ISO 639-3 language code (e.g., 'FRA').
    """
    print(f"👉 Running parsing to syllables and phones for {language}. This may take a while...")

    # Load tokenized text (assuming this is orthographic text, not IPA)
    #input_path = f"produced_data/{language}/{language}_original_sentences.pkl"
    input_path = load_config(language, 'Sentence Data')
    
    if not os.path.exists(input_path):
        print(f"Language corpus '{input_path}' for {language} does not exist.")
        return
    else: print(f"Tokenized language corpus for {language} found at {input_path}")
    
    # Load the tok data
    with open(input_path, "r", encoding="utf-8") as f:
        text = [line.strip().split() for line in f if line.strip()]

    tokenizer, model = load_charsiu_model()

    phonemized_data = []  # list of lists for phonemes
    syllabized_data = []  # list of lists for syllables 

    print(f"Language: {language}")

    for sentence in tqdm(text, desc=f"🔄 Processing:", ncols=80):
        sentence_phonemes = []
        sentence_syllables = []

        for word in sentence:
            # Phone tokenization
            phones = phone_tokenization(word, language)
            if phones:
                sentence_phonemes.append(phones)

            # Syllable tokenization 
            syllables = syllable_tokenization(word, get_onsets_ipa(language), language, tokenizer, model)
            if syllables:
                sentence_syllables.append(syllables)

        phonemized_data.append(sentence_phonemes)
        syllabized_data.append(sentence_syllables)

    # Save data
    folder = f"produced_data/{language}"
    os.makedirs(folder, exist_ok=True)

    pho_output_path = f"{folder}/phonized_{language}.json"
    sylls_output_path = f"{folder}/syllabified_{language}.json" 

    with open(pho_output_path, 'w', encoding='utf-8') as f:
        json.dump(phonemized_data, f, ensure_ascii=False, indent=2)

    with open(sylls_output_path, 'w', encoding='utf-8') as f:
        json.dump(syllabized_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Phonemization completed. Data saved to {pho_output_path}.")
    print(f"✅ Syllabification completed. Data saved to {sylls_output_path}.")




def prepare_text_for_onsets(language):
    """
    Loads tokenized sentences for a language, converts each word to IPA,
    cleans the IPA output, and flattens the sentences into a list of words.
    Saves the resulting list to a pickle file.
    """
    tokenizer, model = load_charsiu_model()

    # Input path
    input_path = f"produced_data/{language}/{language}_original_sentences.pkl"
    if not os.path.exists(input_path):
        print(f"Language corpus '{input_path}' for {language} does not exist.")
        return

    # Load tokenized text (list of list of words)
    with open(input_path, "rb") as f:
        sentences = pickle.load(f)

    # Convert each word in each sentence to cleaned IPA
    ipa_sentences = []
    for sentence in sentences:
        sentence_ipa = []
        for word in sentence:
            cleaned_ipa, _ = generate_ipa(word, language, tokenizer, model)
            sentence_ipa.append(cleaned_ipa)
        ipa_sentences.append(sentence_ipa)

    # Flatten the list of sentences into a single list of words
    flat_ipa_words = [word for sentence in ipa_sentences for word in sentence if word]
    print(f"Total words processed: {len(flat_ipa_words)}")

    # Save the processed text
    output_dir = f"produced_data/{language}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{language}_ipa_sentences.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(flat_ipa_words, f)

    print(f"Onsets for {language} saved to '{output_file}'")


def syllable_tokenization(word, onsets, language, tokenizer, model):
    """
    Performs automatic syllabification of a word using IPA transcriptions and language-specific phonotactic constraints.
    """
    cleaned_ipa, _ = generate_ipa(word, language, tokenizer, model)
    
    if not cleaned_ipa:
        print(f"Failed to generate IPA for word: {word}")
        return []

    phones = phone_tokenization(cleaned_ipa)
    syllables = []
    
    # Define IPA vowels. This list is comprehensive for standard French.
    ipa_vowels = 'iyɪʏeøɛœæaɶɨʉɘɵəɜɞɐɯuʊɤoʌɔɑɒɑ̃ɛ̃œ̃ɔ̃ẽãũĩ'
    is_vowel = lambda p: p in ipa_vowels

    current_pos = 0 # start of the segment we're analyzing for the current syllable

    while current_pos < len(phones):
        # 1. Find the next vowel (nucleus)
        # This vowel will be the nucleus of the syllable we are currently forming.
        vowel_nucleus_idx = -1
        for i in range(current_pos, len(phones)):
            if is_vowel(phones[i]):
                vowel_nucleus_idx = i
                break
        
        if vowel_nucleus_idx == -1:
            # If no more vowels are found, any remaining consonants attach as a coda
            # to the *last* syllable. This handles word-final consonant clusters correctly.
            if syllables and phones[current_pos:]:
                syllables[-1] += ''.join(phones[current_pos:])
            break # No more syllables to form

        # Initialize parts of the current syllable
        current_syllable_parts = []

        # 2. From the phones before the vowel, find the maximal legal onset for the current syllable.
        # These are the consonants from `current_pos` up to `vowel_nucleus_idx - 1`.
        # We try to form the longest possible legal onset from the right side of this segment.
        potential_onset_segment = phones[current_pos : vowel_nucleus_idx]
        current_syllable_onset = ""
        current_syllable_onset_len = 0
        
        # Iterate from the longest possible onset (up to 4 phonemes is usually sufficient)
        # down to a single phoneme.
        for k in range(min(len(potential_onset_segment), 4), 0, -1):
            temp_onset = ''.join(potential_onset_segment[-k:])
            if temp_onset in onsets:
                current_syllable_onset = temp_onset
                current_syllable_onset_len = k
                break
        
        # 3. Remaining consonants before the onset form the coda of the *previous* syllable.
        # If this is the first syllable being formed, there's no previous syllable to attach to.
        # If `current_pos` is not 0, it means we are past the first syllable. Any consonants
        # that could NOT be part of the `current_syllable_onset` must belong to the coda
        # of the *preceding* syllable.
        
        # Calculate the actual start of the current syllable's onset within `phones`
        actual_onset_start_idx = vowel_nucleus_idx - current_syllable_onset_len

        # If there are consonants between `current_pos` and `actual_onset_start_idx`,
        # these are the coda for the *previous* syllable.
        if syllables and current_pos < actual_onset_start_idx:
            syllables[-1] += ''.join(phones[current_pos : actual_onset_start_idx])
            
        # Add the determined onset to the current syllable
        current_syllable_parts.extend(phones[actual_onset_start_idx : vowel_nucleus_idx])
        
        # Add the vowel nucleus to the current syllable
        current_syllable_parts.append(phones[vowel_nucleus_idx])

        # 4. From the phones *after* the current vowel, find the maximal legal onset for the *next* syllable.
        # We need to find the next vowel to define the inter-vocalic consonant cluster.
        next_vowel_search_start = vowel_nucleus_idx + 1
        next_vowel_idx = -1
        for i in range(next_vowel_search_start, len(phones)):
            if is_vowel(phones[i]):
                next_vowel_idx = i
                break

        if next_vowel_idx == -1:
            # If no next vowel, all remaining consonants after the current vowel
            # become the coda of the current syllable. This covers word-final codas.
            current_syllable_parts.extend(phones[next_vowel_search_start:])
            syllables.append(''.join(current_syllable_parts))
            current_pos = len(phones) # Move pointer to end of word
            break # Exit loop, no more syllables

        # Consonants segment between the current vowel and the next vowel.
        # This is where the crucial V1-C*C-V2 split happens.
        inter_vocalic_consonants = phones[next_vowel_search_start : next_vowel_idx]
        
        maximal_next_onset = ""
        maximal_next_onset_len = 0
        
        # Find the maximal legal onset that can begin the *next* syllable.
        # This onset comes from the end of the `inter_vocalic_consonants` segment.
        for k in range(min(len(inter_vocalic_consonants), 4), 0, -1):
            possible_onset = ''.join(inter_vocalic_consonants[-k:])
            if possible_onset in onsets:
                maximal_next_onset = possible_onset
                maximal_next_onset_len = k
                break
        
        # 5. Consonants between the current vowel and the maximal onset of the next syllable
        # form the coda of the current syllable.
        # These are the consonants from the beginning of `inter_vocalic_consonants`
        # up to the point where the `maximal_next_onset` begins.
        coda_for_current_syllable = inter_vocalic_consonants[:len(inter_vocalic_consonants) - maximal_next_onset_len]
        current_syllable_parts.extend(coda_for_current_syllable)

        # Add the completed current syllable to the list
        syllables.append(''.join(current_syllable_parts))
        
        # Update `current_pos` to the beginning of the next syllable's onset
        current_pos = next_vowel_idx - maximal_next_onset_len 

    # TODO: Implement Sonority Sequencing Principle for languages with predictable phonotactics
    # This would typically be applied for finer-grained decisions where Maximal Onset might be ambiguous,
    # or for languages where SSP is the primary rule (e.g., Italian for Coda formation).

    return syllables


def sonori_syllabify(stressed_ipa):
    '''
    See https://github.com/henchc/syllabipy/blob/master/syllabipy/sonoripy.py
    '''
    #TODO: implement fallback for languages which phonotactics are highly predictable using the Sonority Sequencing Principle
    # E.g. Italian, Spanish, Hindi and Arabic
    return []


def get_onsets_ipa(language, threshold=.0002):
    '''
    Takes text in ipa and yields list of onsets and words

    This function is adapted from syllabipy's getOnsets function.
    See https://github.com/henchc/syllabipy/blob/master/syllabipy/legalipy.py
    '''

    input_file = f"produced_data/{language}/{language}_ipa_sentences.pkl"

    # Check if the input file exists, if not, create it
    if not os.path.exists(input_file):
        prepare_text_for_onsets(language)

    # Load the list of IPA words
    with open(input_file, "rb") as f:
        text = pickle.load(f)
    
    # All valid ipa vowels
    # see https://www.internationalphoneticassociation.org/content/ipa-vowels
    ipa_vowels = 'iyɪʏeøɛœæaɶɨʉɘɵəɜɞɐɯuʊɤoʌɔɑɒɑ̃ɛ̃œ̃ɔ̃ẽãũĩ'

    onsets = []
    for word in text:
        word = word.lower()
        onset = ""
        word_phones = phone_tokenization(word)
        for phone in word_phones:
            if phone not in ipa_vowels:  # onset is everying up to first vowel
                onset += phone
            else:
                break
        if onset:  # exclude empty onsets
            onsets.append(onset)

    onsets = [x for x in onsets if x != '']  # get rid of empty onsets

    # now remove onsets caused by errors, i.e. less than .02% of onsets
    freq = Counter(onsets)
    total_onsets = sum(freq.values())

    max_onset_length = 3

    # Keep only frequent, vowel-free, short onsets
    onsets = []
    filtered_onsets = [
        o for o, v in freq.items()
        if (v / total_onsets) > threshold
        and all(char not in ipa_vowels for char in o)
        and len(o) <= max_onset_length
    ]

    # print(f"example onsets: {filtered_onsets[:10]}")
    return filtered_onsets



def phone_tokenization(word): 
    # Unicode grapheme cluster matcher for phones tokenization
    grapheme_pattern = re.compile(r'\X', re.UNICODE)

    # Tokenize into phones
    phones = [match.group() for match in grapheme_pattern.finditer(word) if match.group() not in (' ', '')]

    return phones



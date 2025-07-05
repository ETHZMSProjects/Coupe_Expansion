import os
import regex as re
import pickle
from collections import Counter
from process_ipa import load_charsiu_model, parallelize_ipa_generation, merge_diphthongs
from config_loader import load_config
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def syllable_tokenization_wrapper(args):
    ipa, onsets = args
    return syllable_tokenization(ipa, onsets)

# --- Main Processing Function ---
def parse_to_phones_and_sylls(language):
    """
    Parses sentences to phonemes and syllables using phone_tokenization
    and CharsiuG2P for syllabification.

    Args:
        language (str): The ISO 639-3 language code (e.g., 'FRA').
    """

    # Load tokenized text (assuming this is orthographic text, not IPA)
    #input_path = f"produced_data/{language}/{language}_original_sentences.pkl"
    input_path = load_config(language, 'Sentence Data')
    
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing required language corpus for {language}: {input_path}")
        
        tqdm.write("📥 Loading corpus data ...")
        text = [line.strip().split() for line in Path(input_path).read_text(encoding='utf-8').splitlines() if line.strip()]
        text = text[:100]
        print(f"raw text: {text[:20]}")

    except FileNotFoundError as e:
        logging.error(e)
        raise 

    if len(text) == 0:
        raise ValueError("No data loaded. Input file may be empty or incorrectly formatted.")
    else: tqdm.write(f"✅ Loaded {len(text)} sentences from {input_path}")


    tokenizer, model = load_charsiu_model()

    phonemized_data = []  # list of lists for phonemes
    syllabized_data = []  # list of lists for syllables 
    
    ipa_sentences = parallelize_ipa_generation(text, language, tokenizer, model)

    # Save ipa sentences
    folder = f"produced_data/{language}"
    ipa_output_path = f"{folder}/ipa_corpus_{language}.pkl"
    with open(ipa_output_path, "wb") as f:
        pickle.dump(ipa_sentences, f)

    # Get syllable boundaries for that language
    onsets = get_onsets_ipa(language)

    for ipa_sentence in tqdm(ipa_sentences, desc="🔄 Splitting data to phones and syllables", ncols=80):
        sentence_phones = []
        sentence_syllables = []
        
        inputs = [(ipa, onsets) for ipa in ipa_sentence]

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(syllable_tokenization_wrapper, inputs))

        # One list of phones and one of syllables for each sentence
        sentence_phones = [r[0] for r in results]
        sentence_syllables = [r[1] for r in results]

        phonemized_data.append(sentence_phones)
        syllabized_data.append(sentence_syllables)

    # Post-processing
    if language in ['ENG']:
        syllabized_data = [
            [merge_diphthongs(word) for word in sentence]
            for sentence in syllabized_data
    ]
        
    # Save results
    os.makedirs(f"{folder}/phones", exist_ok=True)
    os.makedirs(f"{folder}/sylls", exist_ok=True)

    print(f"ipa: {ipa_sentences[:20]} ")
    print(f"phonemized data: {phonemized_data[:20]}")
    print(f"syllabized data: {syllabized_data[:20]}")

    pho_output_path = f"{folder}/phones/phonized_{language}.pkl"
    sylls_output_path = f"{folder}/sylls/syllabified_{language}.pkl"

    with open(pho_output_path, "wb") as f:
        pickle.dump(phonemized_data, f)

    with open(sylls_output_path, "wb") as f:
        pickle.dump(syllabized_data, f)

    print(f"✅ Phonemization completed. Data saved to {pho_output_path}.")
    print(f"✅ Syllabification completed. Data saved to {sylls_output_path}.")



def syllable_tokenization(cleaned_ipa, onsets):
    """
    Performs automatic syllabification of a word using IPA transcriptions and language-specific phonotactic constraints.
    """

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

    return phones, syllables


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

    folder = f"produced_data/{language}"
    ipa_path = f"{folder}/ipa_corpus_{language}.pkl"

    with open(ipa_path, "rb") as f:
        ipa_sentences = pickle.load(f)

    # Flatten the list of sentences into a single list of words
    text = [word for sentence in ipa_sentences for word in sentence if word]
    
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






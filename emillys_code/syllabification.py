import os
import regex as re
import pickle
from collections import Counter
from process_ipa import load_charsiu_model, parallelize_ipa_generation, merge_diphthongs
from config_loader import load_config
from tqdm import tqdm
import logging
from pathlib import Path
from joblib import Parallel, delayed

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
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    ipa_output_path = f"{folder}/ipa_corpus_{language}.pkl"
    with open(ipa_output_path, "wb") as f:
        pickle.dump(ipa_sentences, f)
    
    # Get syllable boundaries for that language
    onsets = get_onsets_ipa(language)
    logging.info(f"onsets: {onsets}")

    for ipa_sentence in tqdm(ipa_sentences, desc="🔄 Splitting data to phones and syllables", ncols=80):
        inputs = [(ipa, onsets) for ipa in ipa_sentence]  # build argument pairs
        results = Parallel(n_jobs=-1)(
            delayed(syllable_tokenization_wrapper)(args) for args in inputs
        )
        
        phones, sylls = zip(*results)
        phonemized_data.append(list(phones))
        syllabized_data.append(list(sylls))

    if any(any(sublist) for sublist in ipa_sentences): 
        # Save results
        os.makedirs(f"{folder}/phones", exist_ok=True)
        os.makedirs(f"{folder}/sylls", exist_ok=True)

        logging.info(f"text: {text[:20]}")
        logging.info(f"ipa: {ipa_sentences[:20]}")
        logging.info(f"phonemized data: {phonemized_data[:20]}")
        logging.info(f"syllabized data: {syllabized_data[:20]}")
        
        with open(f"{folder}/phonized_{language}.pkl", "wb") as f:
            pickle.dump(phonemized_data, f)
        
        with open(f"{folder}/syllabified_{language}.pkl", "wb") as f:
            pickle.dump(syllabized_data, f)

        logging.info(f"✅ Tokenization into phones and syllables completed. Data saved to {folder}.")

    else: logging.warning(f"⚠️ Failed to parse to phones and syllables")



def syllable_tokenization(cleaned_ipa, onsets):
    """
    Performs automatic syllabification of a word using IPA transcriptions and language-specific phonotactic constraints.
    """

    phones = phone_tokenization(cleaned_ipa)
    if not phones:
        logging.warning(f"Phone tokenization failed for IPA input: {cleaned_ipa}")
        return [], []

    sylls_prep = merge_diphthongs(phones)
    syllables = []

    current_pos = 0 # start of the segment we're analyzing for the current syllable

    while current_pos < len(sylls_prep):
        # 1. Find the next vowel (nucleus)
        # This vowel will be the nucleus of the syllable we are currently forming.
        vowel_nucleus_idx = -1
        for i in range(current_pos, len(sylls_prep)):
            if is_vowel(sylls_prep[i]):
                vowel_nucleus_idx = i
                break
        
        if vowel_nucleus_idx == -1:
            # If no more vowels are found, any remaining consonants attach as a coda
            # to the *last* syllable. This handles word-final consonant clusters correctly.
            if syllables and sylls_prep[current_pos:]:
                syllables[-1] += ''.join(sylls_prep[current_pos:])
            break # No more syllables to form

        # Initialize parts of the current syllable
        current_syllable_parts = []

        # 2. From the phones before the vowel, find the maximal legal onset for the current syllable.
        # These are the consonants from `current_pos` up to `vowel_nucleus_idx - 1`.
        # We try to form the longest possible legal onset from the right side of this segment.
        potential_onset_segment = sylls_prep[current_pos : vowel_nucleus_idx]
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
            syllables[-1] += ''.join(sylls_prep[current_pos : actual_onset_start_idx])
            
        # Add the determined onset to the current syllable
        current_syllable_parts.extend(sylls_prep[actual_onset_start_idx : vowel_nucleus_idx])
        
        # Add the vowel nucleus to the current syllable
        current_syllable_parts.append(sylls_prep[vowel_nucleus_idx])

        # 4. From the phones *after* the current vowel, find the maximal legal onset for the *next* syllable.
        # We need to find the next vowel to define the inter-vocalic consonant cluster.
        next_vowel_search_start = vowel_nucleus_idx + 1
        next_vowel_idx = -1
        for i in range(next_vowel_search_start, len(sylls_prep)):
            if is_vowel(sylls_prep[i]):
                next_vowel_idx = i
                break

        if next_vowel_idx == -1:
            # If no next vowel, all remaining consonants after the current vowel
            # become the coda of the current syllable. This covers word-final codas.
            current_syllable_parts.extend(sylls_prep[next_vowel_search_start:])
            syllables.append(''.join(current_syllable_parts))
            current_pos = len(sylls_prep) # Move pointer to end of word
            break # Exit loop, no more syllables

        # Consonants segment between the current vowel and the next vowel.
        # This is where the crucial V1-C*C-V2 split happens.
        inter_vocalic_consonants = sylls_prep[next_vowel_search_start : next_vowel_idx]
        
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

    if not syllables:
        fallback = ''.join(sylls_prep) if sylls_prep else 'Ø'
        syllables = [fallback]

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
    
    ipa_vowels = set([
        # Common monophthongs
        'i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ɛ', 'œ', 'æ', 'a', 'ɶ', 'ɨ', 'ʉ',
        'ɘ', 'ɵ', 'ə', 'ɜ', 'ɞ', 'ɐ', 'ɯ', 'u', 'ʊ', 'ɤ', 'o', 'ʌ', 'ɔ', 'ɑ', 'ɒ',

        # Nasal vowels
        'ɑ̃', 'ɛ̃', 'œ̃', 'ɔ̃', 'ẽ', 'ã', 'ũ', 'ĩ'
    ])

    onsets = []
    for word in text:
        word = word.lower()
        phones = phone_tokenization(word)
        
       # NEW safer onset extraction
        vowel_index = next((i for i, ph in enumerate(phones) if ph in ipa_vowels), len(phones))
        onset = ''.join(phones[:vowel_index])
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
    if not filtered_onsets:
        logging.warning(f"No valid onsets extracted for language {language}.")
        return []

    return filtered_onsets


def phone_tokenization(word): 
    # Unicode grapheme cluster matcher for phones tokenization
    grapheme_pattern = re.compile(r'\X', re.UNICODE)

    # Tokenize into phones
    segments = [match.group() for match in grapheme_pattern.finditer(word) if match.group() not in (' ', '')]

    AFFRICATES = {
    'tʃ', 'dʒ', # English
    'tʃ', 'dʒ', 't͡s', 'pf',  # German
    'tʃ', 'dʒ',  # French (not native but appear)
    }
    
    phones = []
    i = 0
    while i < len(segments):
        if i + 1 < len(segments):
            affricate_candidate = segments[i] + segments[i + 1]
            if affricate_candidate in AFFRICATES:
                phones.append(affricate_candidate)
                i += 2
                continue
        # Default case: treat as individual phone
        phones.append(segments[i])
        i += 1

    return phones

# Diacritics and length marker pattern
IPA_DIACRITICS = re.compile(r"[ˈˌːˑ˥˦˧˨˩́̀̂̃̄̆̇]")

# Vowel components
MONOPHTHONGS = {
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
    'ɪ', 'ʏ', 'ʊ',
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
    'ɛ', 'œ', 'ə', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
    'æ', 'ɐ', 'a', 'ɶ', 'ɑ', 'ɒ'
}

DIPHTHONGS = {
    'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ', 'ɪə', 'ɛə', 'ʊə','əʊ',
    'ɥi', 'wi', 'wa', 'wɛ',  # French/English
    'ai', 'au', 'ei', 'ou', 'oi', 'ui',  # Orthographic-style diphthongs
    'ju', 'jə', 'je', 'jʊ', 'wi', 'we', 'wo', 'wə'  # Glide + vowel combos
}

NASAL_VOWELS = {
    'ɑ̃', 'ɛ̃', 'œ̃', 'ɔ̃',  # Standard French
    'ẽ', 'ã', 'ũ', 'ĩ', 'õ'  # Alt notations
}

TRIPHTHONGS = {
    'aɪə', 'aʊə', 'eɪə', 'oʊə', 'ɔɪə'
}

ALL_VOWELS = MONOPHTHONGS | DIPHTHONGS | NASAL_VOWELS | TRIPHTHONGS

GLIDES = {'j', 'w', 'ɥ'}

def strip_diacritics(phone):
    """Remove IPA diacritics and length markers."""
    return IPA_DIACRITICS.sub('', phone)

def is_glide_vowel_combo(phone):
    """Check for sequences like 'ju', 'wə', 'ɥi'."""
    return (
        len(phone) >= 2 and
        phone[0] in GLIDES and
        phone[1:] in MONOPHTHONGS.union(DIPHTHONGS)
    )

def is_vowel(phone):
    """
    Returns True if phone is a vowel (monophthong, diphthong, triphthong, nasal, long vowel, or glide-vowel).
    """
    cleaned = strip_diacritics(phone)

    if cleaned in ALL_VOWELS:
        return True

    if cleaned.endswith('ː') and cleaned[:-1] in ALL_VOWELS:
        return True

    if is_glide_vowel_combo(cleaned):
        return True

    return False






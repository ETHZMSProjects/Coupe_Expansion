import os
import regex as regex_unicode
import pickle
from collections import Counter
from ipa_conversion import load_charsiu_model, parallelize_ipa_generation
from process_ipa import merge_diphthongs, is_vowel
from tqdm import tqdm
import logging
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
import os
import re
import psutil


logging.basicConfig(level=logging.INFO)

def syllable_tokenization_wrapper(args):
    ipa, onsets, language = args
    return syllable_tokenization(ipa, onsets, language)

# --- Main Processing Function ---
def parse_to_phones_and_sylls(language, config_dict):
    """
    Parses sentences to phonemes and syllables using phone_tokenization
    and CharsiuG2P for syllabification.

    Args:
        language (str): The ISO 639-3 language code (e.g., 'FRA').
    """

    input_path = config_dict['Sentence Data']
    folder = f"produced_data/{language}"
    
    # Step 1: Check if IPA corpus already exists, if not generate it
    largest_corpus_size = config_dict['Corpus Size']
    exists, is_near_expected, existing_path = get_largest_ipa_corpus(language, largest_corpus_size)
    
    if exists:
        tqdm.write(f"✅ IPA corpus for {language} exists at {existing_path}. Skipping IPA generation.")
        if not is_near_expected: 
            tqdm.write(f"IPA corpus has limited size. Expected/largest size:{largest_corpus_size}.")
        with open(existing_path, "rb") as f:
            ipa_sentences = pickle.load(f)
            logging.info(f"📊 Memory after loading pickle: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

    # Step 2: If no corpus exists, generate it
    else: 
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

        # Convert corpus to IPA 
        tokenizer, model = load_charsiu_model() # for fallback ipa generation using CharsiuG2P
        parallel_ipa = partial(parallelize_ipa_generation, language=language, tokenizer=tokenizer, model=model, config_dict=config_dict)
        ipa_sentences = parallel_ipa(text)

        # Save ipa sentences
        num_sent = len(ipa_sentences) # ipa corpus size 
        tqdm.write(f"📦 Saving IPA corpus for {language} with {num_sent} sentences ...")
        os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
        ipa_output_path = f"{folder}/ipa_corpus_{language}_size:{num_sent}.pkl"
        existing_path = ipa_output_path
        with open(ipa_output_path, "wb") as f:
            pickle.dump(ipa_sentences, f)
    
    # Step 3: Tokenize IPA sentences into phones and syllables

    # Check whether tokenization for this ipa_corpus was already done

    # Extract the number of sentences from the existing ipa corpus
    match = re.search(r'_size:(\d+)\.pkl$', str(existing_path))
    corpus_size_str = match.group(1) if match else "unknown"

    # Construct expected output paths
    phonized_path = Path(folder) / "phones" / f"phonized_{language}_size:{corpus_size_str}.pkl"
    syllabified_path = Path(folder) / "sylls" / f"syllabified_{language}_size:{corpus_size_str}.pkl"

    # Skip if both files already exist
    if phonized_path.exists() and syllabified_path.exists():
        tqdm.write(f"⏩ Skipping tokenization: phonemized and syllabified data already exist for {language} with size {corpus_size_str}.")
        return existing_path, phonized_path, syllabified_path, corpus_size_str, is_near_expected
    
    # Tokenize into phones and syllables
    else: 

        phonemized_data, syllabized_data = [], []

        # Get syllable boundaries for that language
        onsets = get_onsets_ipa(language, existing_path)

        tqdm.write("🔄 Splitting data into phones and syllables ...")
        # Flatten all words from all sentences for batch processing
        all_words = []
        sentence_word_counts = []
        
        for ipa_sentence in ipa_sentences:
            sentence_word_counts.append(len(ipa_sentence))
            all_words.extend(ipa_sentence)
        
        # Process ALL words at once in parallel
        inputs = [(ipa, onsets, language) for ipa in all_words]
        results = Parallel(n_jobs=15, batch_size=100)(
            delayed(syllable_tokenization_wrapper)(args) for args in inputs
        )
        
        # Reconstruct sentence structure
        word_idx = 0
        
        for sentence_length in sentence_word_counts:
            sentence_phones = []
            sentence_sylls = []
            
            for _ in range(sentence_length):
                phones, sylls = results[word_idx]
                sentence_phones.append(phones)
                sentence_sylls.append(sylls)
                word_idx += 1
                
            phonemized_data.append(sentence_phones)
            syllabized_data.append(sentence_sylls)

        if any(any(sublist) for sublist in ipa_sentences): 
            # Save results
            os.makedirs(f"{folder}/phones", exist_ok=True)
            os.makedirs(f"{folder}/sylls", exist_ok=True)

            #logging.info(f"text: {text[:20]}")
            #logging.info(f"ipa: {ipa_sentences[:20]}")
            
            logging.info(f"phonemized data: {phonemized_data[:20]}")
            logging.info(f"syllabized data: {syllabized_data[:20]}")
            
            with open(phonized_path, "wb") as f:
                pickle.dump(phonemized_data, f)

            with open(syllabified_path, "wb") as f:
                pickle.dump(syllabized_data, f)

            logging.info(f"✅ Tokenization into phones and syllables completed. Data saved to {folder}.")

        else: logging.warning(f"⚠️ Failed to parse to phones and syllables")

    return existing_path, phonized_path, syllabified_path, corpus_size_str, is_near_expected


def syllable_tokenization(cleaned_ipa, onsets, language):
    """
    Performs automatic syllabification of a word using IPA transcriptions and language-specific phonotactic constraints.
    """

    phones = phone_tokenization(cleaned_ipa)
    if not phones:
        logging.warning(f"Phone tokenization failed for IPA input: {cleaned_ipa}")
        return [], []

    sylls_prep = merge_diphthongs(phones, language)
    syllables = []

    current_pos = 0 # start of the segment we're analyzing for the current syllable

    while current_pos < len(sylls_prep):
        # 1. Find the next vowel (nucleus)
        # This vowel will be the nucleus of the syllable we are currently forming.
        vowel_nucleus_idx = -1
        for i in range(current_pos, len(sylls_prep)):
            #print(f"Checking candidate for vowel: {sylls_prep[i]}")
            if is_vowel(sylls_prep[i]):
                #print('vowel found')
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

        # Calculate the actual start of the current syllable's onset within `phones`
        actual_onset_start_idx = vowel_nucleus_idx - current_syllable_onset_len

        # 3. Attach unparsed pre-nuclear consonants to the current syllable
        if not syllables and current_pos < actual_onset_start_idx:
            current_syllable_parts.extend(sylls_prep[current_pos : actual_onset_start_idx])
        
        # 4. Remaining consonants before the onset form the coda of the *previous* syllable.
        # If this is the first syllable being formed, there's no previous syllable to attach to.
        # If `current_pos` is not 0, it means we are past the first syllable. Any consonants
        # that could NOT be part of the `current_syllable_onset` must belong to the coda
        # of the *preceding* syllable.
    

        # If there are consonants between `current_pos` and `actual_onset_start_idx`,
        # these are the coda for the *previous* syllable.
        if syllables and current_pos < actual_onset_start_idx:
            syllables[-1] += ''.join(sylls_prep[current_pos : actual_onset_start_idx])
            
        # Add the determined onset to the current syllable
        current_syllable_parts.extend(sylls_prep[actual_onset_start_idx : vowel_nucleus_idx])
        
        # Add the vowel nucleus to the current syllable
        current_syllable_parts.append(sylls_prep[vowel_nucleus_idx])

        # 5. From the phones *after* the current vowel, find the maximal legal onset for the *next* syllable.
        # We need to find the next vowel to define the inter-vocalic consonant cluster.
        next_vowel_search_start = vowel_nucleus_idx + 1
        next_vowel_idx = -1
        for i in range(next_vowel_search_start, len(sylls_prep)):
            #print(f"Checking candidate for vowel: {sylls_prep[i]}")
            if is_vowel(sylls_prep[i]):
                #print('vowel found')
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


def get_onsets_ipa(language, ipa_corpus_path, threshold=.0001):
    '''
    Takes text in ipa and yields list of onsets and words

    This function is adapted from syllabipy's getOnsets function.
    See https://github.com/henchc/syllabipy/blob/master/syllabipy/legalipy.py

    It extracts onsets from a given language's IPA corpus, ensuring that:
    - It only extracts consonants before the first vowel
    - Onsets do not contain vowels
    - Low-frequency onsets are discarded, reducing noise from mis-syllabified or tokenization errors
    '''

    with open(ipa_corpus_path, "rb") as f:
        ipa_sentences = pickle.load(f)
        logging.info(f"📊 Memory after loading pickle: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

    # Flatten the list of sentences into a single list of words
    # skips invalid words
    text = [word for sentence in ipa_sentences for word in sentence if word]

    onsets = []
    for word in text:
        word = word.lower()
        phones = phone_tokenization(word)
        
        vowel_index = -1
        for i, ph in enumerate(phones):
            #print(f"Checking candidate for vowel: {ph}")
            if is_vowel(ph):
                #print('vowel found')
                vowel_index = i
                break
        
        if vowel_index > 0:  # Only extract if there are consonants before the first vowel
            potential_onset = phones[:vowel_index]
            
            # Double-check that none of the onset phones contain vowels
            if not any(is_vowel(phone) for phone in potential_onset):
                onset = ''.join(potential_onset)
                if onset:  # exclude empty onsets
                    onsets.append(onset)

    onsets = [x for x in onsets if x != '']  # get rid of empty onsets

    # now remove onsets caused by errors, i.e. less than .02% of onsets
    freq = Counter(onsets)
    total_onsets = sum(freq.values())

    #max_onset_length = 3

    # Keep only frequent, vowel-free, short onsets
    onsets = []
    filtered_onsets = [
        o for o, v in freq.items()
        if (v / total_onsets) > threshold
        and not any(is_vowel(char) for char in phone_tokenization(o))
    ]
    if not filtered_onsets:
        logging.warning(f"No valid onsets extracted for language {language}.")
        return []

    return filtered_onsets


def phone_tokenization(word): 
    # Unicode grapheme cluster matcher for phones tokenization
    grapheme_pattern = regex_unicode.compile(r'\X', regex_unicode.UNICODE)

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
        
        # Attach markers to previous segment
        if i + 1 < len(segments) and segments[i+1] in {'ː', 'ˑ'}:
            phones.append(segments[i] + segments[i+1])
            i += 2
            continue
        
        # Default case: treat as individual phone
        phones.append(segments[i])
        i += 1

    return phones


def get_largest_ipa_corpus(language, expected_size):
    folder = Path(f"produced_data/{language}") 
    if not os.path.exists(folder):
        return False, None

    max_size = -1
    best_file = None
    tolerance = 50

    for filename in os.listdir(folder):
        if filename.startswith(f"ipa_corpus_{language}_size:"):
            size_part = filename.split("_size:")[-1].replace(".pkl", "")
            if size_part.isdigit():
                size = int(size_part)
                if size > max_size:
                    max_size = size
                    best_file = filename

    
    if best_file:
        best_path = folder / best_file
        is_near_expected = abs(max_size - expected_size) <= tolerance
        return True, is_near_expected, best_path
    else:
        return False, False, None











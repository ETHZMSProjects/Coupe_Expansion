import re
import regex as re
from num2words import num2words
import jieba
import logging


jieba.setLogLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)


# Compile regex once
DIGIT_RE = re.compile(r'\d+')

def convert_numbers(word_list, language, config_dict): 
    """
    Converts numeric digits in words to their spoken form using num2words,
    only for supported languages and only when digits are present.

    Args:
        word_list (List[str]): List of words to process.
        language (str): ISO language code (e.g., 'ENG').
        config_dict (dict): Dictionary containing num2words language code.

    Returns:
        List[str]: Words with numbers converted to words where applicable.
    """
    num2word_code = config_dict["num2words Code"]

    if language in ["YUE", "CMN", "VIE"]: 
        return word_list  # Skip number conversion for these languages

    def replace_digits(word):
        try:
            return DIGIT_RE.sub(lambda x: num2words(int(x.group()), lang=num2word_code), word)
        except NotImplementedError:
            logging.warning(f"Language '{language}' not supported by num2words. Skipping number conversion.")
            return word
        except Exception as e:
            logging.error(f"Error converting number in word '{word}': {e}")
            return word

    # Only process words that contain digits
    return [replace_digits(w) if any(c.isdigit() for c in w) else w for w in word_list]


def merge_clitics(tokens, language):
    if language == "FRA":
        merged = []
        skip = False
        for i, token in enumerate(tokens):
            if skip:
                skip = False
                continue
            if token.endswith("'") and i + 1 < len(tokens):
                merged.append(token + tokens[i + 1])
                skip = True
            else:
                merged.append(token)
        return merged
    else:
        return tokens 
    
# Diacritics and length marker pattern
IPA_DIACRITICS = re.compile(r"[ˈˌːˑ˥˦˧˨˩̯́̀̂̃̄̆̇]")
    
def strip_diacritics(phone):
    """Remove IPA diacritics and length markers."""
    return IPA_DIACRITICS.sub('', phone)

def is_glide_vowel_combo(phone, glides, monophthongs, diphthongs):
    base = strip_diacritics(phone)
    return (
        len(base) >= 2 and
        base[0] in glides and
        base[1:] in monophthongs.union(diphthongs)
    )

def is_vowel(phone):
    """
    Returns True if phone is a vowel (monophthong, diphthong, triphthong, nasal, long vowel, or glide-vowel).
    """
    base = strip_diacritics(phone)

    monophthongs, glides, nasal_vowel_components = get_monophthong_sets()
    triphthongs, core_diphthongs, glide_vowel_diphthongs, vowel_glide_diphthongs, german_umlauts, french_nasals = get_diphthong_sets()
    
    DIPHTHONGS = core_diphthongs | glide_vowel_diphthongs | vowel_glide_diphthongs | german_umlauts | french_nasals
    
    ALL_VOWELS = monophthongs | DIPHTHONGS | nasal_vowel_components | triphthongs

    return (
        base in ALL_VOWELS or # Check if it's in our predefined vowel sets
        is_glide_vowel_combo(base, glides, monophthongs, DIPHTHONGS) # Check for glide-vowel combinations
    )

def match_with_or_without_marker(candidate, diphthong_set):
    return candidate in diphthong_set or strip_diacritics(candidate) in diphthong_set
    
def merge_diphthongs(phones, language):
    """
    Merges diphthongs and triphthongs in a list of IPA phones into single tokens for syllabification.
    Handles English, German, and French diphthongs/triphthongs with proper prioritization.
    
    Prioritization:
    1. Triphthongs (3 phones) first
    2. Core diphthongs (vowel + vowel) over glide + vowel
    3. Glide + vowel combinations
    
    Args:
        phones (List[str]): A list of phone-level IPA.
    
    Returns:
        List[str]: A new list with diphthongs/triphthongs merged as single items.
    """
    
    triphthongs, core_diphthongs, glide_vowel_diphthongs, vowel_glide_diphthongs, german_umlauts, french_nasals = get_diphthong_sets()
    
    merged = []
    i = 0

    # Update core diphthongs based on language
    if language.upper() == "FRA":
        core_diphthongs.update(french_nasals)
    elif language.upper() == "DEU":
        core_diphthongs.update(german_umlauts)

    
    while i < len(phones):
        matched = False
        
        # Check triphthongs first (highest priority)s
        if i + 2 < len(phones):
            candidate_triphtong = phones[i] + phones[i + 1] + phones[i + 2]
            #print(f"Checking candidate: {candidate_triphtong}")
            if match_with_or_without_marker(candidate_triphtong, triphthongs):
                #print(f"matched")
                merged.append(candidate_triphtong)
                i += 3
                matched = True
                continue
        
        # Before merging any diphthong, check if there's a triphthong starting at i+1
        # that would be broken by merging a diphthong at i
        if not matched and i + 1 < len(phones):
            candidate_diphthong = phones[i] + phones[i + 1]
            
            # Check if merging this diphthong would prevent a triphthong at i+1
            should_skip_for_triphthong = False
            if i + 3 < len(phones):
                potential_triphthong = phones[i + 1] + phones[i + 2] + phones[i + 3]
                if potential_triphthong in triphthongs:
                    should_skip_for_triphthong = True
            
            if not should_skip_for_triphthong:
                # Priority 1: Core diphthongs (vowel + vowel)
                #print(f"Checking candidate: {candidate_diphthong}")
                if match_with_or_without_marker(candidate_diphthong, core_diphthongs):
                    #print('matched')
                    merged.append(candidate_diphthong)
                    i += 2
                    matched = True
                # Priority 2: Glide + vowel
                    #print(f"Checking candidate: {candidate_diphthong}")
                elif match_with_or_without_marker(candidate_diphthong, glide_vowel_diphthongs):
                    #print('matched')
                    merged.append(candidate_diphthong)
                    i += 2
                    matched = True
                # Priority 3: Vowel + glide
                    #print(f"Checking candidate: {candidate_diphthong}")
                elif match_with_or_without_marker(candidate_diphthong, vowel_glide_diphthongs):
                    #print('matched')
                    merged.append(candidate_diphthong)
                    i += 2
                    matched = True
        
        # If no match found, keep the single phone
        if not matched:
            merged.append(phones[i])
            i += 1
    
    return merged

def get_monophthong_sets():
    
    monophtongs = {
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
    'ɪ', 'ʏ', 'ʊ',
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
    'ɛ', 'œ', 'ə', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
    'æ', 'ɐ', 'a', 'ɶ', 'ɑ', 'ɒ'
    }

    glide_components = {'j', 'w', 'ɥ'}
    
    nasal_vowel_components = {
        'ɑ̃', 'ɛ̃', 'œ̃', 'ɔ̃',  # Standard French
        'ẽ', 'ã', 'ũ', 'ĩ', 'õ'  # Alt notations
    }

    return monophtongs, glide_components, nasal_vowel_components
    
    
def get_diphthong_sets():

        
    # TRIPHTHONGS: Vowel + Glide + Schwa/Rhotic (highest priority)
    triphthongs = {
        # English triphthongs
        'aɪə', 'aʊə', 'eɪə', 'oʊə', 'ɔɪə',
        'aɪɚ', 'aʊɚ', 'eɪɚ', 'oʊɚ', 'ɔɪɚ',
        # English alternative realizations
        'juə', 'jʊə', 'jɪə',
        # French (in gliding speech)
        'waɪ', 'waj', 'ɥij', 'ɥiə',
        # German potential triphthongs
        'aɪə', 'aʊə', 'ɔɪə'  # in unstressed contexts
    }

    # CORE DIPHTHONGS: Vowel + Vowel combinations (second priority)
    core_diphthongs = {
        # English
        'eɪ', 'aɪ', 'ai', 'ɔɪ', 'aʊ', 'oʊ',
        'ɪə', 'ɛə', 'ʊə', 'ɑə', 'ɔə',
        'iə', 'uə', 'eə', 'əʊ',
        'ɪɚ', 'ɛɚ', 'ʊɚ', 'ɔɚ', 'aɚ', 'ɚə',

        # Standard German diphthongs
        'aɪ', 'aʊ', 'ɔɪ', 
        
        # Vowel + schwa combinations 
        'iə', 'eə', 'uə', 'oə', 'øə', 'yə', 'ɔə', 'ɛə', 'ɪə', 'ʊə',
        
        # Vowel + /ɐ/ combinations (German r-vocalization)
        'iɐ', 'eɐ', 'uɐ', 'oɐ', 'øɐ', 'yɐ', 'ɔɐ', 'ɛɐ', 'ɪɐ', 'ʊɐ', 'aɐ',

        # French
        'ei', 'ɛi', 'ɔi', 'ui', 'øi', 'ie', 'ye', 'ue',
        'au', 'eu', 'ɛu', 'ou', 'ɔu', 'œu', 'iu', 'io', 'ɑə',

        # Common vowel-vowel across languages
        'ɪi', 'ʊu', 'ɛe', 'ɔo', 'aə', 'əa'
    }

    # GLIDE + VOWEL DIPHTHONGS: j/w/ɥ + Vowel (third priority)
    glide_vowel_diphthongs = {
        # Standard combinations
        'ja', 'je', 'ji', 'jo', 'ju', 'jɑ', 'jɛ', 'jɪ', 'jɔ', 'jʊ', 'jə', 'jɚ',
        'wa', 'we', 'wi', 'wo', 'wu', 'wɑ', 'wɛ', 'wɪ', 'wɔ', 'wʊ', 'wə', 'wɚ',
        'ɥa', 'ɥe', 'ɥi', 'ɥo', 'ɥu', 'ɥy', 'ɥø', 'ɥœ', 'ɥɛ', 'ɥɔ', 'ɥɑ',
        
        # German-specific glide combinations
        'jy', 'jø', 'jœ', 'wy', 'wø', 'wœ',
        'jɐ', 'wɐ',  # with r-vocalization
        
    }

    # VOWEL + GLIDE DIPHTHONGS: Vowel + j/w (lowest priority)
    vowel_glide_diphthongs = {
        # Standard combinations
        'aj', 'ej', 'ij', 'oj', 'uj', 'ɑj', 'ɛj', 'ɪj', 'ɔj', 'ʊj', 'əj', 'ɚj',
        'aw', 'ew', 'iw', 'ow', 'uw', 'ɑw', 'ɛw', 'ɪw', 'ɔw', 'ʊw', 'əw', 'ɚw',
        
        # German-specific
        'øj', 'yj', 'œj', 'ɐj', 'ɐw',
        'øw', 'yw', 'œw',
        
        # Complex umlaut combinations
        'øyj', 'œyj', 'øyw', 'œyw'
    }

    german_umlaut_diphthongs = {
        # øy and variants (like 'øy' in "Freunde", "neu", etc.)
        'øy', 'øʏ', 'œy', 'œʏ',
        # Reversed 
        'yø', 'ʏø', 'yœ', 'ʏœ', 'yɪ', 'ʏɪ',
        
        # others
        'ɔø', 'øɔ', 'ɛœ', 'œɛ',
        
        # With reduced vowels
        'øə', 'œə', 'yə', 'ʏə'
    }

    
    # 2. FRENCH NASALIZED VOWEL DIPHTHONGS 
    french_nasal_diphthongs = {
        # Nasalized vowel + vowel
        'ɑ̃i', 'ɑ̃u', 'ɛ̃i', 'ɛ̃u', 'œ̃i', 'œ̃u', 'ɔ̃i', 'ɔ̃u',
        # Vowel + nasalized vowel  
        'iɑ̃', 'uɑ̃', 'iɛ̃', 'uɛ̃', 'iœ̃', 'uœ̃', 'iɔ̃', 'uɔ̃',
        # Alternative notations
        'ãi', 'ãu', 'ẽi', 'ẽu', 'ĩa', 'ũa', 'õi', 'õu'
    }
    
    
    return triphthongs, core_diphthongs, glide_vowel_diphthongs, vowel_glide_diphthongs, german_umlaut_diphthongs, french_nasal_diphthongs





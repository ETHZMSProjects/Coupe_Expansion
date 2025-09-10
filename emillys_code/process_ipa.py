import re
import regex as re
from num2words import num2words
import jieba
import logging


jieba.setLogLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)


# Compile regex once
DIGIT_RE = re.compile(r'\d+')

def convert_numbers(word_list: list[str], language: str, config_dict: dict) -> list[str]:
    """
    Convert digit substrings to spoken forms using `num2words`, language-aware.

    Parameters
    ----------
    word_list : list of str
        Tokens to process; only tokens containing ASCII digits are considered.
    language : str
        ISO-3 language code used to gate conversion (e.g., 'ENG', 'FRA').
        Conversion is skipped for {"YUE", "CMN", "VIE"}.
    config_dict : dict
        Must contain:
          - "num2words Code": str, language code understood by `num2words`
            (e.g., 'en', 'fr', 'de').

    Returns
    -------
    list of str
        Same-length list where digit substrings are replaced by their
        language-specific names when supported; otherwise unchanged.

    Notes
    -----
    • Robust to `num2words` NotImplementedError: falls back to original token.
    • Non-digit tokens are not touched (fast path).
    """
    try: 
        num2word_code = config_dict["num2words Code"]
    except KeyError:
        logging.error(f"num2words Code not found in config for language '{language}'. Skipping number conversion.")
        return word_list

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


def merge_clitics(tokens: list[str], language: str) -> list[str]:
    """
    Merge simple clitic + host patterns in tokenized text (language-specific).

    Parameters
    ----------
    tokens : list of str
        Token sequence (already segmented on whitespace/punctuation).
    language : str
        ISO-3 language code. Currently:
          • 'FRA': merges sequences like ["l'", "ami"] → ["l'ami"].
          • Other languages: no change.

    Returns
    -------
    list of str
        Token sequence with language-specific clitics merged.

    Notes
    -----
    • This is a minimal heuristic (apostrophe-final token + following token).
    """
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
    
def strip_diacritics(phone: str) -> str:
    """
    Remove IPA diacritics and length markers from a phone symbol.

    Parameters
    ----------
    phone : str
        IPA phone (may include primary/secondary stress, length, tone/diacritics).

    Returns
    -------
    str
        Base phone with characters in `IPA_DIACRITICS` removed.

    Notes
    -----
    • Preserves segmental letters; removes markers like ˈ, ˌ, ː, tone diacritics.
    """
    return IPA_DIACRITICS.sub('', phone)

def is_glide_vowel_combo(
    phone: str,
    glides: set[str],
    monophthongs: set[str],
    diphthongs: set[str]
) -> bool:
    """
    Test if a phone string matches a glide + vowel combination.

    Parameters
    ----------
    phone : str
        Candidate phone sequence (diacritics already stripped or not).
    glides : set of str
        Set of glide symbols (e.g., {'j', 'w', 'ɥ'}).
    monophthongs : set of str
        Base vowel symbols considered monophthongs.
    diphthongs : set of str
        Vowel sequences considered diphthongs (for membership checks).

    Returns
    -------
    bool
        True if `phone` begins with a glide and the remainder is a vowel
        (monophthong or diphthong), False otherwise.

    Notes
    -----
    • Internally applies `strip_diacritics` before matching.
    """
    base = strip_diacritics(phone)
    return (
        len(base) >= 2 and
        base[0] in glides and
        base[1:] in monophthongs.union(diphthongs)
    )

def is_vowel(phone: str) -> bool:
    """
    Determine whether an IPA phone is vowel-like (mono/di/triphthong, nasalized, or glide–vowel).

    Parameters
    ----------
    phone : str
        IPA phone (may include diacritics/length markers).

    Returns
    -------
    bool
        True if `phone` belongs to monophthongs, diphthongs, triphthongs,
        nasal-vowel sets, or forms a valid glide–vowel combo.

    Notes
    -----
    • Uses `get_monophthong_sets()` and `get_diphthong_sets()` to form
      comprehensive vowel inventories; strips diacritics before lookup.
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

def match_with_or_without_marker(candidate: str, diphthong_set: set[str]) -> bool:
    """
    Match a candidate sequence against a diphthong inventory with diacritic tolerance.

    Parameters
    ----------
    candidate : str
        Candidate IPA string (may include length/diacritic markers).
    diphthong_set : set of str
        Canonical diphthong strings without diacritics/length.

    Returns
    -------
    bool
        True if `candidate` is in `diphthong_set` either as-is or after
        `strip_diacritics(candidate)`, else False.
    """
    return candidate in diphthong_set or strip_diacritics(candidate) in diphthong_set
    
def merge_diphthongs(phones: list[str], language: str) -> list[str]:
    """
    Merge diphthongs and triphthongs into single tokens with priority rules.

    Priority (applied left-to-right):
      1) Triphthongs (3-segment sequences)
      2) Core diphthongs (vowel + vowel)
      3) Glide + vowel diphthongs
      4) Vowel + glide diphthongs

    Parameters
    ----------
    phones : list of str
        IPA phone sequence to merge.
    language : str
        ISO-3 language code; augments core diphthongs for:
          • 'FRA' → nasal vowel sequences
          • 'DEU' → umlaut-related sequences

    Returns
    -------
    list of str
        New sequence where recognized (tri)diphthongs are merged.

    Notes
    -----
    • Prevents breaking a triphthong starting at i+1 by merging a diphthong at i.
    • Compares both raw strings and diacritic-stripped forms.
    • Does not alter consonant clusters.
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

def get_monophthong_sets() -> tuple[set[str], set[str], set[str]]:
    """
    Return canonical monophthongs, glides, and nasalized vowel components.

    Returns
    -------
    tuple of sets
        (
          monophthongs,
          glide_components,
          nasal_vowel_components
        )

    Notes
    -----
    • Monophthong inventory follows standard IPA chart coverage for high→low,
      front→back vowels.
    • Nasal vowel components include French-style nasalized vowels and common
      alternate notations.
    """
    
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
    
    
def get_diphthong_sets() -> tuple[set[str], set[str], set[str], set[str], set[str], set[str]]:
    """
    Return curated sets for tri/diphthongs across ENG/DEU/FRA (and common patterns).

    Returns
    -------
    tuple of sets
        (
          triphthongs,
          core_diphthongs,
          glide_vowel_diphthongs,
          vowel_glide_diphthongs,
          german_umlaut_diphthongs,
          french_nasal_diphthongs
        )

    Notes
    -----
    • Triphthongs prioritize glide/schwa/rhotic realizations typical in English;
      includes limited gliding French forms and permissive German variants.
    • Core diphthongs are vowel+vowel (incl. schwa/ɐ sequences and r-vocalization).
    • Glide–vowel and vowel–glide sets include j/w/ɥ transitions and language-specific
      additions (umlauts for German).
    """

        
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





import logging
import os
from functools import partial
from itertools import product
from pathlib import Path
import sys
import json, os, tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
import numpy as np
from IPython.display import display

from syllabification import parse_to_phones_and_sylls
from info_rate import count_ling_units

logging.basicConfig(level=logging.INFO)

def load_config(language: str) -> Dict[str, Any]:
    """
    Load per-language configuration from language_config.json.

    Expects a JSON file "language_config.json" in the current working directory
    with a column "Language" used as the unique key per row.

    Parameters
    ----------
    language : str
        ISO-style language code (e.g., "FRA", "DEU").

    Returns
    -------
    Dict[str, Any]
        Dictionary view of the configuration row for `language`.

    Raises
    ------
    FileNotFoundError
        If the JSON file is missing.
    ValueError
        If the "Language" column is missing or non-unique.
    KeyError
        If `language` is not present in the config.
    """
    config_path = Path("language_config.json")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    df = pd.read_json(config_path)

    if "Language" not in df.columns:
        raise ValueError('Expected a "Language" column in language_config.json')

    # Ensure uniqueness; if not unique, either drop duplicates or raise
    if not df["Language"].is_unique:
        # choose: raise (strict) or keep="first" (lenient). Here we raise.
        raise ValueError('Column "Language" must be unique per row.')

    df = df.set_index("Language", drop=True)

    try:
        row = df.loc[language]   # Series
    except KeyError:
        raise KeyError(f"Language '{language}' is not supported in language_config.json.")

    # Force str keys to satisfy Dict[str, Any]
    config: Dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}
    return config


def validate_structure(
    data: Any,
    text_type: str,
    processing_type: str,
    context: str = "root",
    nesting: Optional[List[type]] = None,
    ) -> None:  
    """
    Recursively validate the nested structure of processed corpora.

    Many downstream estimators assume a nested list structure of the form
    ``sentences → words → units`` (where *units* are strings of phones/syllables)
    or, for word-level processing, ``sentences → words`` (strings).

    This validator raises informative :class:`TypeError` exceptions if the data
    do not conform to the expected shape.

    Parameters
    ----------
    data : Any
        Object to validate.
    text_type : str
        Text granularity ("across_sentences" or "within_words").
    processing_type : str
        One of {"sylls", "phones", "words"}.
    context : str, optional
        Used to annotate error messages with the position inside the
        nested structure. Defaults to "root".
    nesting : list[type], optional
        Expected nesting level inferred from processing_type

    Raises
    ------
    ValueError
        If "processing_type" is not recognized.
    TypeError
        If any level of "data" does not match the expected container type.


    Examples
    --------
    >>> validate_structure([[['p','a']]], 'across_sentences', 'phones')
    """

    if nesting is None:
        if processing_type in ("sylls", "phones"):
            nesting = [list, list, list, str]  # sentences → words → units (strings)
        elif processing_type == "words":
            nesting = [list, list, str]        # sentences → words (strings)
        else:
            raise ValueError(f"Invalid processing type: {processing_type}")

    expected_type = nesting[0]

    # Check type at this level
    if not isinstance(data, expected_type):
        raise TypeError(f"{context} expected {expected_type.__name__}, got {type(data).__name__}")

    # If more levels remain, recurse
    if len(nesting) > 1:
        for i, elem in enumerate(data):
            validate_structure(elem, text_type, processing_type, f"{context}[{i}]", nesting[1:])



def update_values_in_csv(
    language_to_update: str,
    value: Union[float, List[float]],
    n: int,
    value_type: str,
    text_type: str,
    processing_type: str,
    round_digits: int = 3,
) -> None:
    """
    Insert or update model outputs for a language in:
    - produced_data_large_corpus/inter_intra_comparison.csv
      (within/between text-type comparisons per language and speaker)

    It creates missing columns on-the-fly and appends missing language rows by
    borrowing speaker/passage metadata from '../InfoRateData.csv'.

    Parameters
    ----------
    language_to_update : str
        Language code whose rows should be updated.
    value : float or list[float]
        Either a scalar value applied to all speaker rows of the language, or a
        list whose length matches the number of speakers/passages for that
        language.
    n : int
        N-gram order. Must be one of {1, 2, 3, 4}.
    value_type : {"ID", "IR", "SR"}
        Metric to update.
    text_type : {"within_words", "across_sentences"}
        Text-context granularity.
    processing_type : {"phones", "sylls", "words"}
    round_digits : int, default=3
        Decimal digits for rounding the stored values.

    Raises
    ------
    ValueError
        If n not in {1,2,3,4}, value_type invalid, or list-valued `value`
        length mismatches the number of rows to update.
    """
    # Validate n-gram label
    model = {1: "unigram", 2: "bigram", 3: "trigram", 4: "quadgram"}.get(n)
    if not model:
        raise ValueError("Invalid n value. Only 1, 2, 3, or 4 are allowed.")

    # Validate value_type
    allowed_types = {"ID", "IR", "SR"}
    if value_type not in allowed_types:
        raise ValueError(f"Invalid value_type. Use one of {allowed_types}.")

    # Construct the target column for inter/intra comparison
    inter_intra_column = f"{text_type}_{processing_type}_{value_type}_{model}"

    # File paths
    base_path = "produced_data_large_corpus"
    inter_intra_file = os.path.join(base_path, "inter_intra_comparison.csv")

    # Load CSV
    inter_intra_df = pd.read_csv(inter_intra_file)

    # Ensure target column exists
    if inter_intra_column not in inter_intra_df.columns:
        inter_intra_df[inter_intra_column] = np.nan

    # Load original metadata to add missing language rows (Speaker/Text)
    original_df = pd.read_csv("../InfoRateData.csv", sep="\t")
    inter_intra_languages = inter_intra_df["Language"].tolist()

    # Add rows for this language if missing
    if language_to_update not in inter_intra_languages:
        filtered_speaker_data = original_df[original_df["Language"] == language_to_update]
        speaker_passages = filtered_speaker_data[["Speaker", "Text"]].drop_duplicates()

        new_rows = []
        for _, row_data in speaker_passages.iterrows():
            row = {
                "Speaker": row_data["Speaker"],
                "Language": language_to_update,
                "Text": row_data["Text"],
            }
            # Fill all existing columns; set to NaN unless provided above
            for col in inter_intra_df.columns:
                if col not in row:
                    row[col] = np.nan
            new_rows.append(row)

        inter_intra_df = pd.concat([inter_intra_df, pd.DataFrame(new_rows)], ignore_index=True)

    # Mask + indices for this language
    inter_mask = inter_intra_df["Language"] == language_to_update
    inter_indices = inter_intra_df[inter_mask].index

    # Write values
    if isinstance(value, list):
        if len(value) != len(inter_indices):
            raise ValueError(
                "Length of value list must match number of speakers/passages for the language in inter_intra_comparison."
            )
        for idx, val in zip(inter_indices, value):
            inter_intra_df.at[idx, inter_intra_column] = round(float(val), round_digits)
    else:
        val = round(float(value), round_digits)
        inter_intra_df.loc[inter_mask, inter_intra_column] = val

    # Save back
    inter_intra_df.to_csv(inter_intra_file, index=False)



def ask_question(question: str, function_to_run: Callable[[str], Any], language: str) -> bool:
    """
    Prompt the user with a yes/no question and conditionally execute a function.

    Behavior
    --------
    - Prints the question and waits for user input.
    - Accepts "y" or "yes" (case-insensitive) to run the given function.
    - Any other input is treated as "no".
    - If input is unavailable (EOF/KeyboardInterrupt), it defaults to "no"
      and returns False safely without crashing.

    Parameters
    ----------
    question : str
        The text prompt shown to the user.
    function_to_run : Callable[[str], Any]
        A function to execute if the answer is "yes".
        It must accept a single argument: the language string.
    language : str
        Argument forwarded to `function_to_run`.

    Returns
    -------
    bool
        True if the function was executed (answer was "yes"),
        False otherwise.

    """
    def _run():
        function_to_run(language)
        return True

    try:
        answer = input(f"{question} [y/n]: ").strip().lower()
        print(f"> {answer}")
    except (EOFError, KeyboardInterrupt):
        print("⚠️ No input available (EOF/interrupt). Defaulting to 'no'.")
        return False

    if answer in {"y", "yes"}:
        return _run()

    return False

def check_data_availability(
    language: str,
    unit_type: str,
    config_dict: Dict[str, Any],
    folder_name: str, 
    corpus_size: Union[int, str],
) -> Optional[Path]:
    """
    Verify presence of required data and accompanying unit-count metadata for a language.

    The function looks for a preprocessed pickle file corresponding to the chosen
    unit_type and checks whether a companion CSV with linguistic unit counts includes
    complete entries for the target language. If required data are missing or incomplete,
    the user is prompted (interactively) to generate them using:
      - parse_to_phones_and_sylls  (for phones/sylls/words corpora)
      - count_ling_units           (for the counts CSV)

    Parameters
    ----------
    language : str
        Language code (e.g., "FRA").
    unit_type : {"phones", "sylls", "words"}
        Target unit type to validate/generate.
    config_dict : dict
        Language-specific configuration forwarded to the generation functions.
    folder_name : str
        Root folder for language-specific data (per-language data under <folder_name>/<language>).
    corpus_size : int or "max"
        Corpus size to (re)generate when needed.

    Returns
    -------
    Path or None
        Path to the prepared data file if both the pickle and the counts CSV are available
        and pass basic validation; otherwise None.

    Raises
    ------
    ValueError
        If unit_type is not one of {"phones", "sylls", "words"}.

    Notes
    -----
    - Data files are expected under:
        * words:  <folder_name>/<language>/
        * phones: <folder_name>/<language>/phones/
        * sylls:  <folder_name>/<language>/sylls/
    - File name patterns:
        * words:  ipa_corpus_{LANG}_size:*.pkl
        * phones: phonized_{LANG}_size:*.pkl
        * sylls:  syllabified_{LANG}_size:*.pkl
    - The counts CSV is expected at: semantically_similar_texts/ling_units_counts.csv
      with columns {"language", "n_words"} or {"language", "n_phones"} or {"language", "n_syllables"}.
    - This function prompts the user (once) via ask_question; it is not suitable for
      fully automated, non-interactive pipelines.
    """

    # --- Validate unit type and choose filename stem ---
    if unit_type not in ("phones", "sylls", "words"):
        raise ValueError("❌ Invalid processing type. Use 'phones', 'words' or 'sylls'.")

    # --- Paths: generate under <folder_name>/<language>, lookup under .../<unit_type> ---
    lang_base = Path(folder_name) / language
    if unit_type == "words":
        # ipa_corpus files live directly under <folder_name>/<language>
        lookup_dir = lang_base
        stem = "words"
        unit_col = "language" # "n_words" is not in the CSV; presence of language suffices
        pattern = f"ipa_corpus_{language}_size:*.pkl"
    elif unit_type == "phones":
        lookup_dir = lang_base / "phones"
        stem = "phonized"
        unit_col = "n_phones"
        pattern = f"{stem}_{language}_size:*.pkl"
    elif unit_type == "sylls":
        lookup_dir = lang_base / "sylls"
        stem = "syllabified"
        unit_col = "n_syllables"
        pattern = f"{stem}_{language}_size:*.pkl"
    else:
        raise ValueError("❌ Invalid processing type. Use 'phones', 'words' or 'sylls'.")
   

    # Ensure lookup directory exists (so glob doesn't fail on missing parents)
    lookup_dir.mkdir(parents=True, exist_ok=True)

    # --- Search for prepared data file ---
    matches = list(lookup_dir.glob(pattern))


    # --- Generate if missing ---
    if not matches:
        print(f"❌ No prepared {unit_type} data found for {language} at {lookup_dir}.")
        sys.stdout.flush()

        gen_fn = partial(
            parse_to_phones_and_sylls,
            config_dict=config_dict,
            folder=str(lang_base),      # pass per-language base, not .../unit_type
            corpus_size=corpus_size,
        )

        if not ask_question(
            f"Run parse_to_phones_and_sylls('{language}') now to generate the required {unit_type} data?",
            gen_fn,
            language,
        ):
            return None

        # Re-scan after generation
        matches = list(lookup_dir.glob(pattern))
        if not matches:
            print(f"⚠️ Still no data found for {unit_type} after generation.")
            return None

    input_path = matches[0]

    # --- Validate unit counts CSV ---
    ling_unit_count_csv_path = Path("semantically_similar_texts/ling_units_counts.csv")
    check_failed = False

    if not ling_unit_count_csv_path.is_file():
        print(f"❌ Linguistic unit counts CSV not found at: {ling_unit_count_csv_path}.")
        sys.stdout.flush()
        check_failed = True
    else:
        df = pd.read_csv(ling_unit_count_csv_path, sep=None, engine="python")
        required_columns = {"language", unit_col}

        if not required_columns.issubset(df.columns) or language not in set(df["language"]):
            print(required_columns)
            print(df.columns)
            print("❌ Required columns or language entry missing in data file.")
            sys.stdout.flush()
            check_failed = True
        else:
            subset = df[df["language"] == language]
            if subset[unit_col].isnull().any():
                print(f"❌ Missing values for '{language}' in required data file.")
                sys.stdout.flush()
                check_failed = True

    # --- Repair counts if needed ---
    if check_failed:
        count_fn = partial(
            count_ling_units,
            config_dict=config_dict,
            folder=str(lang_base),   # keep consistent with generation
        )
        if not ask_question(
            f"Run count_ling_units('{language}') now to compute/update '{unit_col}' in the counts CSV?",
            count_fn,
            language,
        ):
            return None
        elif ling_unit_count_csv_path.is_file():
            # Optionally re-validate here if you’d like.
            check_failed = False

    # --- Final decision ---
    if input_path.exists() and not check_failed:
        return input_path
    elif input_path.exists() and check_failed:
        print("⚠️ Data file exists but unit count validation may have failed.")
        return None
    else:
        logging.error("❌ Data generation failed")
        return None
    

def check_expected_values(df: pd.DataFrame) -> None:
    """
    Checks whether all expected metric columns are present in a results table and identifies any that are empty.

    The function compares the DataFrame’s columns against the full set of expected combinations 
    (text_type × unit_type × n-gram × metric), excluding the invalid within_words × words case. 
    It then reports any missing or extra columns, and for each language highlights columns that 
    exist but contain only missing values (all NaN), which may indicate a failed pipeline stage.

    Parameters
    ----------
    df : pandas.DataFrame
    Results table that includes metadata columns ``{"Speaker","Language","Text"}``
    and metric columns following the project naming convention.

    Returns
    -------
    None
    Prints a human-readable report to stdout.
    """
    # Define expected columns 
    expected_processing_types = ['phones', 'sylls', 'words']
    text_types = ['within_words', 'across_sentences']
    n_values = ['unigram', 'bigram', 'trigram', 'quadgram']
    metrics = ['ID', 'IR', 'SR']

    # Generate expected column names
    expected_columns = {
        f"{tt}_{pt}_{metric}_{n}"
        for tt, pt, n, metric in product(text_types, expected_processing_types, n_values, metrics)
        if not (tt == 'within_words' and pt == 'words')  # Exclude invalid combo
    }

    # Find actual columns (ignoring metadata)
    actual_columns = set(df.columns) - {'Speaker', 'Language', 'Text'}

    # Compare
    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns

    logging.info(f"✅ Total expected columns: {len(expected_columns)}")
    logging.info(f"✅ Total actual columns: {len(actual_columns)}")

    if missing:
        logging.info("\n❌ Missing columns:")
        for col in sorted(missing):
            logging.info(f"  - {col}")
    else:
        logging.info("\n✅ All expected combinations are present.")

    if extra:
        logging.info("\n⚠️ Extra unexpected columns:")
        for col in sorted(extra):
            logging.info(f"  - {col}")

    # Check for entirely empty columns per language
    result = {}
    feature_cols = [col for col in df.columns if col not in ['Speaker', 'Language', 'Text']]
    
    for lang, group in df.groupby('Language'):
        empty_cols = [
            col for col in feature_cols
            if group[col].apply(lambda x: pd.isna(x) or str(x).strip().lower() in {"", "nan", "none"}).all()
        ]

        if empty_cols:
            result[lang] = empty_cols

    if result:
        print("\n⚠️ Columns present but entirely empty (all NaN), per language:")
        for lang, cols in sorted(result.items()):
            print(f"  {lang}:")
            for col in sorted(cols):
                print(f"    - {col}")
    else:
        print("\n✅ No expected columns are entirely empty for any language.")
    

from pathlib import Path
import os

def clean_corpus_size_files(
    base_folder: Union[str, Path],
    language_codes: Sequence[str],
    corpus_size: Union[int, str],
    unit_types: Sequence[str],
    ) -> None:
    """
    Delete generated files matching a given ``size:{corpus_size}`` pattern.

    The function scans each language folder and its processing-type subfolders
    inside base_folder and deletes any files whose names contain size:{corpus_size}. 
    This helps prevent conflicts with outdated data when regenerating corpora 
    at a new target size.

    Parameters
    ----------
    base_folder : str or Path
     Root directory that contains per-language outputs (e.g., ``"produced_data"``).
    language_codes : e.g., ``["DEU", "ENG"]``).
    corpus_size : int or str
        Size token to match (e.g., ``500``, or ``"max"``).
    processing_types : one in {"phones", "sylls", "words}


    Returns
    -------
    None
    Operates for side effects (file deletions) and logs progress.


    Notes
    -----
    The function skips missing folders and reports them with a warning-level log
    entry. Any deletion errors are caught and logged; the routine continues with
    the next file.
    """
    base_folder = Path(base_folder)
    pattern = f"size:{corpus_size}"
    files_deleted = 0

    for language in language_codes:
        lang_folder = base_folder / language

        # First: clean files directly in base_folder / language
        if lang_folder.exists():
            for file in lang_folder.iterdir():
                if file.is_file() and pattern in file.name:
                    try:
                        file.unlink()
                        logging.info(f"🗑️ Deleted {file}")
                        files_deleted += 1
                    except Exception as e:
                        logging.warning(f"❌ Could not delete {file.name}: {e}")
        else:
            logging.warning(f"⚠️ Folder not found: {lang_folder}")

        # Second: clean files in each processing_type subfolder
        for unit_type in unit_types:
            target_folder = lang_folder / unit_type
            if target_folder.exists():
                for file in target_folder.iterdir():
                    if file.is_file() and pattern in file.name:
                        try:
                            file.unlink()
                            logging.info(f"🗑️ Deleted {file}")
                            files_deleted += 1
                        except Exception as e:
                            logging.warning(f"❌ Could not delete {file.name}: {e}")
            else:
                logging.warning(f"⚠️ Folder not found: {target_folder}")

    if files_deleted == 0:
        logging.warning(f"ℹ️ No files matched pattern 'size:{corpus_size}'")


def create_minimal_summary(df_summary: pd.DataFrame, corpus_size: Union[int, str]) -> None:
    """
    Displays a styled HTML table summarizing model outputs.

    Builds a pivot table from a long-form summary DataFrame (with columns 
    ["Language","UnitType","TextType","Metric","n","Value"]). The table shows 
    mean values and applies a consistent color scale across all metrics and 
    n-gram orders to make comparisons easier.

    Parameters
    ----------
    df_summary : pandas.DataFrame
        Long-form table with at least the columns listed above.
    corpus_size : int or str
        Corpus size label used for the printed header.


    Returns
    -------
    None

    Notes
    -----
    - The function does not write to disk
    """
    print(f"\nResults for corpus size {corpus_size}:\n")

    table = df_summary.pivot_table(
        values="Value",
        index=["Language", "UnitType"],
        columns=["TextType", "Metric", "n"],
        aggfunc="mean"
    )

    #  Normalize the color gradient globally across the whole table
    vmin = df_summary["Value"].min()
    vmax = df_summary["Value"].max()

    # Display styled table
    display(
        table.style
        # Center data cells
        .set_properties(subset=pd.IndexSlice[:, :], **{"text-align": "center"})
        # Center headers 
        .set_table_styles([
            {"selector": "th.col_heading", "props": [("text-align", "center !important")]},
            {"selector": "th.row_heading", "props": [("text-align", "center !important")]},
            {"selector": "th.blank", "props": [("visibility", "hidden"), ("padding", "0"), ("width", "0")]},
        ])
        .background_gradient(cmap="Blues", vmin=vmin, vmax=vmax)
        .format("{:.2f}")
    )

def init_worker_logging(
    log_level: str | int = "INFO",
    logfile: str = "logs/run.log"
) -> None:
    """Initialize logging configuration for worker processes.

    This function sets up Python's built-in ``logging`` system so that
    log messages from workers (e.g., when using multiprocessing or joblib)
    are consistently formatted and written both to a file and to the
    console (stderr).

    Args:
        log_level:
            Logging level to use. Can be either a string such as
            ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, or
            ``"CRITICAL"``, or an integer constant from the
            :mod:`logging` module (e.g., ``logging.INFO``).
            Default is ``"INFO"``.
        logfile:
            Path to the log file. If the file already exists, new log
            entries are appended. Default is ``"logs/run.log"``.

    Returns:
        None. The function configures the global logging system in-place.

    Notes:
        - The log format includes the log level, process ID, timestamp,
          logger name, and the message.
        - Both a ``FileHandler`` (writing to ``logfile``) and a
          ``StreamHandler`` (writing to ``sys.stderr``) are installed.
        - The ``force=True`` flag ensures that any existing logging
          configuration is overridden.
    """
    level = getattr(logging, log_level) if isinstance(log_level, str) else log_level
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] [%(process)d] %(asctime)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="a"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )

def create_folder_structure(
    languages: Sequence[str],
    folder_name: str,
    unit_types: Sequence[str]
) -> None:
    """
    Create a hierarchical folder structure for language-specific corpus processing.

    The function ensures that a directory tree exists with the following layout:

        <folder_name>/
            lambdas/
            <language_1>/
                <unit_type_1>/
                <unit_type_2>/
                ...
            <language_2>/
                ...
            ...

    Parameters
    ----------
    languages : Sequence[str]
        A sequence of ISO 3-letter language codes (e.g., ``["FRA", "DEU", "ENG"]``).
        Each code corresponds to a top-level directory containing subfolders
        for different linguistic unit types.
    folder_name : str
        The name of the root directory under which the structure will be created.
        If the directory does not exist, it will be created.
    unit_types : Sequence[str]
        The linguistic unit types (e.g., ``["phones", "sylls", "words"]``)
        for which subfolders will be created under each language directory.

    Returns
    -------
    None
        The function has no return value. It creates directories on disk
        if they do not already exist.

    Notes
    -----
    - All directories are created with ``exist_ok=True`` semantics, i.e.,
      no error is raised if a directory already exists.
    - A common directory named ``lambdas`` is created once at the root level
      of ``folder_name`` to store lambda tuning results or metadata.
    - The function is idempotent: repeated calls will not overwrite or
      remove existing files or directories.

    Examples
    --------
    >>> create_folder_structure(
    ...     languages=["FRA", "DEU", "ENG"],
    ...     folder_name="test_runs",
    ...     unit_types=["phones", "sylls", "words"]
    ... )
    # Produces:
    # test_runs/
    # ├── lambdas/
    # ├── FRA/
    # │   ├── phones/
    # │   ├── sylls/
    # │   └── words/
    # ├── DEU/
    # │   ├── phones/
    # │   ├── sylls/
    # │   └── words/
    # └── ENG/
    #     ├── phones/
    #     ├── sylls/
    #     └── words/
    """
    base = Path(folder_name)
    base.mkdir(parents=True, exist_ok=True)

    # Create lambdas directory once
    (base / "lambdas").mkdir(parents=True, exist_ok=True)

    for lang in languages:
        lang_dir = base / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        for proc in unit_types:
            (lang_dir / proc).mkdir(parents=True, exist_ok=True)


def write_ready_manifest(
    language: str,
    unit_type: str,
    folder_name: str,
    existing_ipa_path=None,
    phonized_path=None,
    syllabified_path=None
) -> Path:
    """
    Write a minimal "ready" manifest file for a completed preprocessing task.

    This function creates a JSON manifest and a simple `.ok` marker in a
    structured task directory. The manifest captures essential metadata
    (task identity, corpus size, input file presence/size) in a machine-
    and human-readable format.

    The manifest file is written atomically to avoid corruption on crashes:
    - A temporary file is written and fsynced.
    - The temp file is then renamed to `READY.manifest.json`.

    A marker file `READY.ok` is also written, enabling quick presence checks
    without parsing JSON.

    Parameters
    ----------
    language : str
        ISO code or identifier for the language (e.g. `"FRA"`, `"DEU"`).
    unit_type : str
        Unit of analysis (`"words"`, `"phones"`, `"sylls"`).
    folder_name : str
        Root folder under which all task results are organized.
    corpus_size_actual : int or str
        Number of items actually included in the corpus (may differ from
        requested size).
    existing_ipa_path : str or Path, optional
        Path to IPA-level word corpus, if available.
    phonized_path : str or Path, optional
        Path to phoneme-level corpus, if available.
    syllabified_path : str or Path, optional
        Path to syllable-level corpus, if available.

    Returns
    -------
    Path
        The path to the written `READY.manifest.json`.

    Manifest Structure
    ------------------
    The JSON manifest has the following structure:

    {
      "created_at": "<UTC ISO timestamp>",
      "status": "ready",
      "task": {
        "language": "FRA",
        "unit_type": "phones"
      },
      "corpus_size_actual": 100,
      "input_paths": {
        "words": {"path": "...", "exists": true, "size_bytes": 1234},
        "phones": {"path": "...", "exists": true, "size_bytes": 5678},
        "sylls": {"path": null, "exists": false, "size_bytes": null}
      }
    }

    Notes
    -----
    - The parent directory structure `<folder>/<language>/<unit_type>/`
      is created automatically if it does not exist.
    - The manifest does not attempt to validate corpus contents, only
      records file metadata (path, existence, size).
    - The `.ok` marker is a plain text file containing `"OK"`, useful
      for quick downstream checks.
    """
    # target paths
    task_dir = Path(folder_name) / language / unit_type 
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "READY.manifest.json"
    ok_path = task_dir / "READY.ok"

    # small helper
    def info(p):
        if p is None:
            return {"path": None, "exists": False, "size_bytes": None}
        p = Path(p)
        return {
            "path": str(p),
            "exists": p.exists(),
            "size_bytes": (p.stat().st_size if p.exists() else None),
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "task": {"language": language, "unit_type": unit_type},
        "input_paths": {
            "words": info(existing_ipa_path),
            "phones": info(phonized_path),
            "sylls": info(syllabified_path),
        }
    }

    # atomic write
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(task_dir), suffix=".tmp", encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, manifest_path)

    # quick marker
    ok_path.write_text("OK\n")

    return manifest_path

import json
from pathlib import Path

def is_task_ready(
    language: str,
    unit_type: str,
    folder_name: str,
    validate_manifest: bool = False,
) -> bool:
    """
    Check whether a (language, unit_type, text_type) task is marked as ready.

    A task is considered "ready" if a READY.ok marker file exists in the
    corresponding directory. If `validate_manifest=True`, the function also
    attempts to read and verify the READY.manifest.json file.

    Parameters
    ----------
    language : str
        Language code (e.g. "FRA", "ENG", "DEU").
    unit_type : str
        Unit of analysis ("words", "phones", "sylls").
    folder_name : str
        Root folder containing all task outputs.
    validate_manifest : bool, default=False
        If True, also check that READY.manifest.json exists, is valid JSON,
        and has `"status": "ready"`.

    Returns
    -------
    bool
        True if the task is ready, False otherwise.

    Examples
    --------
    >>> is_task_ready("FRA", "phones", "across_sentences", "test_runs")
    True

    >>> is_task_ready("ENG", "sylls", "across_sentences", "test_runs")
    False
    """
    task_dir = Path(folder_name) / language / unit_type
    ok_path = task_dir / "READY.ok"
    manifest_path = task_dir / "READY.manifest.json"

    if not ok_path.exists():
        return False

    if validate_manifest:
        if not manifest_path.exists():
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("status") != "ready":
                return False
        except Exception:
            return False

    return True













    

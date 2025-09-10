import logging 
import numpy as np
import pickle
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
import pandas as pd
import argparse

from helpers import update_values_in_csv, is_task_ready, load_config, init_worker_logging
from syllabification import parse_to_phones_and_sylls
from resource_manager import get_n_jobs_config

from ngram_model import JelinekMercerModel

def parse_args():
    p = argparse.ArgumentParser(description="Run JM n-gram IR computation pipeline.")
    p.add_argument("--languages", nargs="+", default=["FRA","DEU","ENG"],
                   help="ISO codes, e.g. FRA DEU ENG")
    p.add_argument("--unit-types", nargs="+", default=["phones", "sylls","words"],
                   choices=["phones","sylls","words"])
    p.add_argument("--text-types", nargs="+", default=["across_sentences"],
                   choices=["across_sentences","within_words"])
    p.add_argument("--n-values", nargs="+", type=int, default=[1,2,3,4])
    p.add_argument("--folder-name", default="produced_data_large_corpus")
    p.add_argument("--corpus-size", default="max", help="'max' or integer")
    p.add_argument("--n-jobs", type=int, default=2,
                   help="Exact number of parallel workers to use (no auto-detect).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    
    # --- RAM controls ---
    p.add_argument("--ram-budget-gb", type=float, default=64.0,
                   help="Upper limit on RAM the pipeline is allowed to consider (GiB).")
    p.add_argument("--gb-per-job", type=float, default=20.0,
                   help="Estimated RAM needed per worker (GiB).")
    p.add_argument("--reserve-frac", type=float, default=0.20,
                   help="Safety headroom fraction to keep free (e.g., 0.20 = 20%).")
    
    return p.parse_args()

def run_pipeline(language, unit_type, text_type, n_values, folder_name, corpus_size):
    logger = logging.getLogger(__name__)   
    folder = Path(folder_name) / language

    try:
        config_dict = load_config(language)

        (
            existing_ipa_path,
            phonized_path,
            syllabified_path,
            corpus_size_str,
            is_near_expected,
        ) = parse_to_phones_and_sylls(
            language=language,
            config_dict=config_dict,
            folder=folder,
            corpus_size=corpus_size,
        )

        # Check whether the preprocessing steps were completed for this task
        if is_task_ready(language, unit_type, folder_name, validate_manifest=True):

            if unit_type == 'words': 
                input_path = existing_ipa_path
            else: 
                input_path = phonized_path if unit_type == 'phones' else syllabified_path

            if not input_path.exists():
                logger.error(f"Input path does not exist: {input_path}")
                return None
            with open(input_path, "rb") as f:
                data = pickle.load(f)
            
            df_rows = []
            for n in n_values:
                jm_model = JelinekMercerModel(n)
                results = jm_model.fit_with_tuning(data, text_type, unit_type, language)

                logger.info(
                f"[n={n} | {text_type} | {unit_type}] "
                f"Best λs (dev-tuned): {results['best_lambdas']} | "
                f"Mean IR: {np.mean(results['info_rate_values']):.3f} | "
                f"Dev PPL={results['dev_perplexity']:.3f} | "
                f"Test PPL={results['test_perplexity']:.3f}"
                )

                df_rows.append({
                    "language": language,
                    "unit_type": unit_type,
                    "text_type": text_type,
                    "n": n,
                    "mean_IR": float(np.mean(results["info_rate_values"])),
                    "mean_ID": float(np.mean(results["info_density"])),
                    "mean_SR": float(np.mean(results["speech_rate_values"])),
                    "dev_perplexity": float(results["dev_perplexity"]),
                    "test_perplexity": float(results["test_perplexity"]),
                })

                # Save the results for a corpus with largest possible size 
                if is_near_expected: 
                    update_values_in_csv(language, results["info_density"], n, 'ID', text_type, unit_type)
                    update_values_in_csv(language, results["info_rate_values"], n, 'IR', text_type, unit_type)
                    update_values_in_csv(language, results["speech_rate_values"], n, 'SR', text_type, unit_type) 
                    
                    # Save the model to a file
                    jm_model.save_model(language, folder, unit_type, text_type, corpus_size_str)
                    del jm_model
                    import gc; gc.collect()

            return pd.DataFrame(df_rows) if df_rows else None
        else:
            logger.warning(f"Preprocessing not completed for {language}/{unit_type}/{text_type}. Skipping. First run all steps in data_prep.ipynb")
            return None
        
    except Exception as e:
        logger.exception(f"Pipeline failed for {language}/{unit_type}/{text_type}: {e}")
        return None

def main():

    #---- Get args -----
    args = parse_args()

    # ---- logs dir ----
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ---- logging initialization ----
    log_level = getattr(logging, args.log_level)  # map "INFO" -> logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] [%(process)d] %(asctime)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "run.log", mode="a"),
            logging.StreamHandler()
        ],
        force=True  # avoid duplicate handlers on re-runs
    )
    logger = logging.getLogger(__name__)

    # ---- tasks ----
    n_jobs = get_n_jobs_config(logger, args)
    all_tasks = product(args.languages, args.unit_types, args.text_types)
    # Exclude impossible task combos up front
    tasks = [
        (lang, proc, txt)
        for (lang, proc, txt) in all_tasks
        if not (proc == "words" and txt == "within_words")
    ]

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10,
        initializer=init_worker_logging,
        initargs=(args.log_level, "logs/run.log"),
        pre_dispatch="1*n_jobs",   # don't queue lots of tasks ahead
        batch_size=1,              # keep tasks tiny
        max_nbytes="1M",           # memmap arrays larger than 1 MB
        mmap_mode="r",             # read-only memmap
        temp_folder="logs/../tmp", # or a path you set; same as $JOBLIB_TEMP_FOLDER
    )(
        delayed(run_pipeline)(lang, proc, txt, args.n_values, args.folder_name, args.corpus_size)
        for (lang, proc, txt) in tasks
    )

    # Count number of successful tasks
    success_count = sum(r is not None for r in results)
    if success_count == 0:
        logger.error("❌ No valid results. Please check the input data and configurations.")
    else:
        logger.info(f"✅ {success_count} tasks completed successfully.")


if __name__ == "__main__":
    main()
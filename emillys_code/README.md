# An Exploration of Measuring Information Rate Across Human Languages  

This repository provides code and data processing pipelines to estimate **information density (ID)**, **speech rate (SR)**, and **information rate (IR)** using **n-gram Markov models**.  
Our contribution extends Coupé et al. (2019) by systematically varying **n-gram order (1–4), linguistic unit (phones, syllables, words), and corpus scope (within words vs. across sentences n-grams)**.  

---

## 🔬 Scientific Motivation  

Human languages differ greatly in speech tempo, phonotactics, and structure, yet evidence suggests they may transmit information at comparable rates.  
This project tests the robustness of that claim:  
- Do IR values remain stable across modeling choices (unit, n-gram order, corpus type)?  
- Or are near-universal IR estimates artifacts of specific methods?  

**Main finding:** IR values vary systematically with modeling choices (phones > syllables > words; decreasing with n-gram order; higher at sentence level than within words). This highlights methodological limits of earlier work and suggests future models should use more naturalistic corpora and long-range neural models.  

---

## 📑 Table of Contents  

- [Repository Overview](#-repository-overview)  
- [Methods](#️-methods)  
  - [Computed Metrics](#-computed-metrics)  
  - [Linguistic Units Supported](#linguistic-units-supported)  
  - [Data Sources](#-data-sources)  
  - [Supported Languages](#-supported-languages)  
  - [Workflows](#-workflows)  
    - [Interactive Data Preparation Pipeline](#1-interactive-data-preparation-pipeline)  
    - [Command-Line Pipeline for Heavy IR Computation](#2-command-line-pipeline-for-heavy-ir-computation)  
- [Usage](#-usage)  
  - [Installation and Dependencies](#1--installation-and-dependencies)  
  - [Set-Up](#set-up) 
- [Visualizations](#-visualizations)  
- [Key References](#-key-references)  
- [How to Cite](#-how-to-cite)  
- [Authors](#-authors)  

---

## 📁 Repository Overview

- `IR_comp_pipeline.ipynb` — Pipeline to compute metrics and update CSV summaries.  
- `ngram_model.py` — Contains the `BaseNgramModel`, `MLEModel`, and `JelinekMercerModel` classes for constructing n-gram models.  
- `helpers.py` — Utility functions.  
- `plotting.ipynb` — Generates plots to visualize information rate trends.  
- `data_prep.ipynb` — Notebook with preprocessing steps to generate the required data.  
- `info_rate.py` — Computes information rate and counts linguistic units in semantically similar sentences.  
- `ipa_conversion.py` — Uses espeak-ng or CharsiuG2P to convert words into International Phonetic Alphabet (IPA).  
- `language_config.json` — Defines language-specific configurations.  
- `process_ipa.py` — Cleans IPA by removing suprasegmental symbols, punctuation, and non-phonemic artifacts, while preserving a configurable set of language-specific IPA characters.  
- `syllabification.py` — Splits words into phones and syllables.  
- `resource_manager.py` — Determines memory limits and safely clamps parallel `n_jobs` to fit within available RAM.  
- `inter_intra_comparison.csv` — Stores per-speaker and per-language metrics (ID, IR, SR) across linguistic units (phones, syllables, words) and text scopes (within-words, across-sentences).  

---

## ⚙️ Methods  

#### 📏 Computed Metrics  

We compute **information density (ID)**, **speech rate (SR)**, and **information rate (IR)**:  

- **Entropy estimation (ID):** Conditional entropy via n-gram models (order 1–4).  
  - **Jelinek–Mercer Smoothing:** interpolation with λ tuned by minimizing development set perplexity.  
- **Speech rate (SR):** Units per second, computed as `SR = NU / D`, where NU is the number of units in a semantically similar text and D is its phonation time (excluding pauses).  
- **Information rate (IR):** Transmission rate in bits per second, computed as `IR = ID × SR`.  
- Final IR estimates per language are obtained by averaging across speakers.  

---

#### Linguistic Units Supported
- Phones (phonetic segments)  
- Syllables  
- Words  

---

#### 📂 Data Sources

- **Written data:** [Tatoeba subtitles corpus (OPUS, v2023-04-12)](https://opus.nlpl.eu/) (tokenized).  
- **Spoken data:** 15 [semantically similar texts](https://github.com/ETHZMSProjects/Coupe_Expansion/tree/master/emillys_code/semantically_similar_texts) from Coupé et al. (2019).  
- **Phonation time:** Duration measures of each speaker reading the semantically similar texts ([AutomaticSylDetect.csv](https://github.com/ETHZMSProjects/Coupe_Expansion/blob/master/AutomaticSylDetect.csv)) from Coupé et al. (2019).  

---

#### 🌍 Supported Languages

| Language | Code | 
|----------|------|
| French   | FRA  |
| German   | DEU  | 
| English  | ENG  |  

---

### 🔄 Workflows  

This repository supports two workflows:  

1. **Interactive Data Preparation Pipeline**  
2. **Command-Line Pipeline for Heavy IR Computation**  

---

#### 1. Interactive Data Preparation Pipeline  

Runs preprocessing and verifies all required data exists (using `corpus_size = "max"`).  
Also useful for sanity checks and debugging with small datasets (e.g. `corpus_size = 100`).  

- Ensures folder structure under `<folder_name>/<LANG>/{phones,sylls}/`.  
- Calls `check_ready_to_run(...)` and `check_data_availability(...)` to verify pickle and CSV files:  
  - Pickles in `<folder>/<LANG>/{phones,sylls}/` with the largest corpus size.  
  - Counts CSV at `semantically_similar_texts/ling_units_counts.csv` with required columns:  
    - Phones → `{"language", "n_phones"}`  
    - Sylls → `{"language", "n_syllables"}`  
- If missing/incomplete, prompts generation via `parse_to_phones_and_sylls(...)` or `count_ling_units(...)`.  
- Writes a manifest and `.ok` file to indicate all required data is present.  
- Produces a summary table via `create_minimal_summary(...)` for quick checks.  

**Preprocessing steps:**  
1. Convert words into IPA using [espeak-ng](https://github.com/espeak-ng/espeak-ng) and [CharsiuG2P](https://arxiv.org/abs/2204.03067).  
2. Normalize text (Unicode NFC), convert digits to words, and merge clitics.  
3. Clean IPA by removing suprasegmentals, punctuation, and artifacts.  
4. Apply syllabification (Maximal Onset Principle with language-specific legal onsets).  
5. Count linguistic unit occurrences in semantically similar texts.  

---

#### 2. Command-Line Pipeline for Heavy IR Computation  

For each `(language, unit_type, text_type)` task, this pipeline:  

- Checks the manifest to confirm preprocessing is complete.  
- Runs preprocessing via `parse_to_phones_and_sylls(...)` if needed.  
- Trains **Jelinek–Mercer smoothed n-gram models** for each `n` in `--n-values`:  
  - Tunes λ on development perplexity.  
  - Logs best λ, mean IR, and perplexity scores.  
  - Computes metrics per `n`: **ID, IR, SR, dev_perplexity, test_perplexity**.  
- If using `corpus_size = "max"`:  
  - Updates the metrics CSV via `update_values_in_csv(...)`.  
  - Saves models via `jm_model.save_model(...)`.  

##### File Structure & Naming
```
produced_data_large_corpus/<LANG>/
  phones/phonized_<LANG>_size:*.pkl
  sylls/syllabified_<LANG>_size:*.pkl
  ipa_corpus_<LANG>_size:*.pkl     # words live directly under <LANG>/
```
---

## 🚀 Usage

### 1. 🖥 Installation and Dependencies  

- **Python ≥ 3.10**  
- **Core dependencies** (installed automatically via `requirements.txt`)  
- **External tools** (must be installed separately):  
  - [**espeak-ng**](https://github.com/espeak-ng/espeak-ng) – used via subprocess calls for phoneme-level conversion.  
  - [**CharsiuG2P**](https://arxiv.org/abs/2204.03067) – pretrained grapheme-to-phoneme model (loaded via HuggingFace/Transformers).  
  - [**num2words**](https://github.com/savoirfairelinux/num2words) – number-to-text conversion (installed automatically).  

#### Set-Up:

```bash
git clone https://github.com/ETHZMSProjects/Coupe_Expansion.git
cd Coupe_Expansion/emillys_code
pip install -r requirements.txt
```

2. Download the written **corpus** (see *Data Sources* above).
3. Obtain **phonotation time** measures (see *Data Sources* above).
4. Specify the path to the written corpus under `"Sentence Data"` in `language_config.json`(e.g. `"../../data/CMN/cmn.tok"`).
5. Adjust `corpus_size` in `language_config.json` if needed.
6. Run `run_test_pipeline()` in `data_prep.ipynb` to prepare all required data.
7. For heavy runs, use `IR_comp_pipeline.py`.  

**Running `IR_comp_pipeline.py`:**

Set parameters in your terminal and run.
e.g.
```
python IR_comp_pipeline.py \
  --languages FRA DEU ENG \
  --unit-types phones sylls words \
  --text-types across_sentences within_words \
  --n-values 1 2 3 4 \
  --folder-name produced_data_large_corpus \
  --corpus-size max \
  --n-jobs 2 \
  --log-level INFO \
  --ram-budget-gb 64.0 \
  --gb-per-job 20.0 \
  --reserve-frac 0.40
```

**Notes and Constraints**

- Valid `--unit-types`: `phones`, `sylls`, `words`  
- Valid `--text-types`: `across_sentences`, `within_words`  
- Invalid: `words` + `within_words` (excluded automatically)  
- `--corpus-size`: `"max"` or an integer (e.g. `100` for testing)  
- IR computation may take >24h for syllables/words with higher `n`  
- Parallelism is capped using RAM heuristics (see `resource_manager.py`)  
- Use `corpus_statistics.ipynb` to inspect corpus statistics (STTR, number of n-grams, unit counts, etc.).  

---

## 📈 Visualizations  

Run `plotting.ipynb` to generate:  

- Comparison tables for IR/ID/SR across languages, units, and n-grams (`get_IR_ID_SR_table(...)`).  
- Plots of information rate as a function of n-gram order (`plot_inforate_vs_ngram_order(...)`).  
- Combined plots comparing multiple text levels (`plot_inforate_combined_texttypes(...)`).  

---

## 📚 Key References

- C. Coupé, Y. M. Oh, D. Dediu, and F. Pellegrino, *Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche.* Science Advances, 5(9), eaaw2594, 2019.  
- C. E. Shannon, *Prediction and entropy of printed English.* Bell System Technical Journal, 30(1), 50–64, 1951. [Link](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1951.tb01366.x)  
- Y. M. Oh, *Linguistic complexity and information: Quantitative approaches.* PhD thesis, Université de Lyon, 2015. [Link](https://theses.fr/2015LYO20072)  
- J. Duddington and R. H. Dunn, *espeak-ng: Open-source speech synthesizer.* GitHub, accessed Sept 2025. [Link](https://github.com/espeak-ng/espeak-ng)  
- J. Zhu, C. Zhang, and D. Jurgens, *ByT5 model for massively multilingual grapheme-to-phoneme conversion.* arXiv:2204.03067, 2022. [Link](https://arxiv.org/abs/2204.03067)  
- Savoir-faire Linux, *num2words: Convert numbers to words in multiple languages.* GitHub, accessed Sept 2025. [Link](https://github.com/savoirfairelinux/num2words)  
- S. F. Chen and J. Goodman, *An empirical study of smoothing techniques for language modeling.* Computer Speech & Language, 13(4), 359–394, 1999.  
---

## 📖 How to Cite

If you use this repository in your research, please cite it using the DOI below:  

[DOI....]
### Plain-text citation
Sidaine-Daumiller, E. (2025). An Exploration of Measuring Information Rate Across Human Languages (Version X.X) [Computer software]. Zenodo. https://doi.org/...

### BibTeX
```bibtex
@software{sidainedaumiller2025inforate,
  author       = {Emilly Sidaine-Daumiller, Jacob Ayers, Julia Ulrich, Richard Hahnloser},
  title        = {An Exploration of Measuring Information Rate Across Human Languages},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {X.X},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```
For reproducibility, please cite the version-specific DOI corresponding to the release you used

---

## 👤 Authors
- **Emilly Sidaine-Daumiller**
- Jacob Ayers
- Julia Ulrich
- Richard Hahnloser <br>

All authors are affiliated with **ETH Zurich and the University of Zurich**

---
                

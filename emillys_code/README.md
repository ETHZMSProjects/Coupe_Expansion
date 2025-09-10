#An Exploration of Measuring Information Rate Across Human Languages 

This repository provides code and data processing pipelines to estimate **information density (ID)**, **speech rate (SR)**, and **information rate (IR)** using **n-gram Markov models**.  

Our contribution extends Coupé et al. (2019) by systematically varying **n-gram order (1–4), linguistic unit (phones, syllables, words), and corpus scope (within words vs. across sentences)**.  

---

## 🔬 Scientific Motivation  

Human languages differ greatly in tempo, phonotactics, and structure, yet evidence suggests they transmit information at comparable rates (~39 bit/s). 
This project tests the robustness of that claim:  
- Do IR values remain stable across modeling choices (unit, n-gram order, corpus type)?  
- Or are near-universal IR estimates artifacts of specific methods?  

**Main finding:** IR values vary systematically with modeling choices (phones > syllables > words; decreasing with n-gram order; higher at sentence level than within words). This highlights methodological limits of earlier work and suggests future models should use more naturalistic corpora and long-range neural models.  

---

## 📂 Data  

### Sources  
- **Written data**: [Tatoeba subtitles corpus (OPUS, v2023-04-12)](https://opus.nlpl.eu/).  
- **Spoken data**: 15 semantically similar texts and phonotation times from [Coupé et al. (2019)]([https://opus.nlpl.eu/](https://github.com/ETHZMSProjects/Coupe_Expansion/tree/master/emillys_code/semantically_similar_texts)).  

### Preprocessing  
- Conversion to **IPA** via [espeak-ng](https://github.com/espeak-ng/espeak-ng) and [CharsiuG2P](https://arxiv.org/abs/2204.03067).  
- Normalization: Unicode NFC, digit-to-words conversion, clitic merging (e.g., French l’homme).  
- Cleaning: remove suprasegmentals, punctuation, artifacts.  
- Syllabification: rule-based **Maximal Onset Principle** with language-specific legal onsets.  

**Linguistic units supported:**  
- Phones (phonetic segments)  
- Syllables  
- Words  

---

## ⚙️ Methods  

We compute **information density (ID)**, **speech rate (SR)**, and **information rate (IR)**:  

- **Entropy estimation**: Conditional entropy via n-gram models (1–4).  
- **Smoothing**: Jelinek–Mercer interpolation (λ tuned by perplexity minimization).  
- **Speech rate**: Units per second, computed from semantically similar texts.  
- **Information rate**:  
  \[
  IR = ID \times SR
  \]  

**Pipeline design:**  
1. Convert corpus → IPA → phones/syllables/words.  
2. Build n-gram models with JM smoothing.  
3. Compute entropy (ID), SR, and IR.  
4. Save results to CSV and visualization notebooks.


 ## 🚀 Usage

This repository supports two workflows:

1. **Interactive Data Preperation Pipeline**
2. **Command Line Pipeline for heavy IR computation**

The pipeline can **generate missing inputs on demand** (interactive prompts) and will **save models and metrics** when the largest corpus size is used.

### 1) Interactive Data Preperation Pipeline
What it does:

Ensures folder structure under produced_data_large_corpus/<LANG>/[phones|sylls]/.

Calls check_ready_to_run() which uses:

check_data_availability() → verifies required pickles and the counts CSV.

Prompts you to generate missing data via:

parse_to_phones_and_sylls(...)

count_ling_units(...)

Produces a small summary table via create_minimal_summary(...) for visual/CSV sanity checks.

Interactive prompts: This test uses ask_question(...) to confirm generation steps. It is not intended for non-interactive runs (cluster jobs, CI).

### 2)
The main CLI accepts language/unit/text settings, model orders, corpus size, parallel workers, logging, and RAM budgeting.

General form:
python your_script.py \
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
  --reserve-frac 0.20

Notes & constraints

Valid --unit-types: phones, sylls, words

Valid --text-types: across_sentences, within_words

Impossible combo: words + within_words is excluded automatically.

--corpus-size: "max" or an integer (e.g., 100 for testing).

Parallelism is capped by get_n_jobs_config(...) using RAM heuristics:

--ram-budget-gb = total memory to consider

--gb-per-job = estimated worker footprint

--reserve-frac = safety headroom fraction (kept free)

Logs: logs/run.log (+ stdout), joblib temp: tmp/ under repo.

What the pipeline does

For each (language, unit_type, text_type) task:

Preprocessing / Inputs

parse_to_phones_and_sylls(...) creates/returns:

existing_ipa_path (words)

phonized_path (phones)

syllabified_path (sylls)

check_data_availability(...) verifies:

Expected pickles under <folder>/<language>/[phones|sylls]/

Counts CSV at semantically_similar_texts/ling_units_counts.csv with required columns:

phones → {"language","n_phones"}

sylls → {"language","n_syllables"}

words → presence of "language" row suffices

If missing/incomplete, you’ll be prompted to generate via parse_to_phones_and_sylls(...) or count_ling_units(...).

Modeling

For each n in --n-values, train Jelinek–Mercer smoothed n-gram models.

Lambdas tuned on dev perplexity; logs show best λ, mean IR, and perplexity scores.

Metrics & Saving

Metrics per n: ID, IR, SR, dev_perplexity, test_perplexity.

If is_near_expected (largest corpus), automatically:

Update CSV via update_values_in_csv(...)

Save model files via jm_model.save_model(...).

### File Structure & Naming
produced_data_large_corpus/<LANG>/
  phones/phonized_<LANG>_size:*.pkl
  sylls/syllabified_<LANG>_size:*.pkl
  ipa_corpus_<LANG>_size:*.pkl     # words live directly under <LANG>/

Counts CSV
 semantically_similar_texts/ling_units_counts.csv
# required columns:
#   phones    -> language, n_phones
#   syllables -> language, n_syllables
#   words     -> language (presence only)

Results / models

Models: produced_data_large_corpus/<LANG>/**

Aggregated metrics CSV updated via update_values_in_csv(...)

---

## 🖥 Installation & Dependencies  

- **Python ≥ 3.10**  
- Dependencies: `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`  
- External tools: [espeak-ng](https://github.com/espeak-ng/espeak-ng), [CharsiuG2P](https://arxiv.org/abs/2204.03067), [num2words](https://github.com/savoirfairelinux/num2words).  


git clone https://github.com/ETHZMSProjects/Coupe_Expansion.git
cd Coupe_Expansion/emillys_code
pip install -r requirements.txt

## Troubleshooting


##  Table of Contents

- [ Repository Structure](#-repository-structure)  
- [ Supported Languages](#-supported-languages)  
- [ Processing Types](#-processing-types)  
- [ Desciption of the Pipeline](#-description-of-the-pipeline)
- [ Computed Metrics](#-computed-metrics)  
- [ Running the Pipeline](#-running-the-pipeline)  
- [ Visualizations](#-visualizations)  
- [ References](#-references)  
- [ Author](#-author)

---

## 📁 Repository Structure

- `markov_chainpipeline.ipynb` — Main pipeline to build Markov models, compute metrics, and update CSV summaries  
- `markov_models.py` — Contains the `MarkovModel` class for constructing and analyzing n-gram models  
- `helpers.py` — Utility functions for data loading, metric calculation, and CSV updates  
- `plotting.ipynb` — Generates plots to visualize Information Rate trends
- `syll_comparison_coupe_esidaine.csv` — Csv file that stores IR and ID calculation for each ngram and author (Coupé and esidaine)

---

## Supported Languages

| Language          | Code | |
|-------------------|------|------|
| French            | FRA  | 🇫🇷   |
| German            | DEU  | 🇩🇪   |
| English           | ENG  | 🇬🇧   |

---

## Desciption of the Pipeline

1. **Load the Data**  
   Parse phonetically transcriped (IPA) and syllabified words for each language.

2. **Build Markov Models**  
   Train 1-gram to 4-gram models  
   Save to `.pkl` for reuse

3. **Compute Metrics**  
   Calculates ID, SR, and IR values per n-gram and language 

4. **Update CSV Summary**  
   Updates the data in the following csv file: `syll_comparison_coupe_esidaine.csv`

---

## 📏 Computed Metrics

This project computes:  <br/> 
<img src="emillys_code/images/formulas.png" alt="Formulas" width="400"/>

---
## 🚀 Running the Pipeline

#### Requirements: 
- Folder with the word data for the given languages
- Installed dependencies (see requirements.txt)

1. Open `markov_chain_pipeline.ipynb` and set your parameters:

```python
language = "FRA"
linguistic_unit = "syllables"
```

2. Update file paths as needed and run all cells. 
The output will automatically update the numbers in `syll_comparison_coupe_esidaine.csv`

---

### Output

- Models saved to: `produced_data/{LANGUAGE}/`
- Calculations saved to: `syll_comparison_coupe_esidaine.csv`


#### Sample Output

```
Training a Markov Model with n = 2:
Information Density: 3.2184
Information Rate: 39.4367
Updated ID_bigram_esidaine
Updated IR_bigram_esidaine

Example probabilities (p(x, y)):
p(('wə3',) -> 'i1') = 0.0931
p(('tə5',) -> 'lə5') = 0.0712
p(('i1',) -> 'Qai4') = 0.0556
✅ Saved 2-gram model to 'produced_data/YUE/'
```

---

## 📈 Visualizations

Run `plotting.ipynb` to generate plots for:

- Information Rate Comparison across Languages  
- Information Rate vs. N-gram Order

## 📖 References

Coupé, C., Oh, Y. M., Dediu, D., & Pellegrino, F. (2019). Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche. Science advances, 5(9), eaaw2594. https://doi.org/10.1126/sciadv.aaw2594 <br/>

Pellegrino, F., Coupé, C., & Marsico, E. (2011). Across-Language Perspective on Speech Information Rate. Language, 87, 539 - 558. <br/>

Oh, Y.M. (2015). Linguistic complexity and information : quantitative approaches. 

---
## 👤 Author
**Emilly Sidaine-Daumiller**  
Master’s student in Neural Systems and Computation at University and ETH Zurich 

---
                

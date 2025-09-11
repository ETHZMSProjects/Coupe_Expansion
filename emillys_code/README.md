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

## ⚙️ Methods  

### Linguistic units supported:
- Phones (phonetic segments)  
- Syllables  
- Words

### 📏 Computed Metrics

We compute **information density (ID)**, **speech rate (SR)**, and **information rate (IR)**:  

- **Entropy estimation (ID)**: Conditional entropy via n-gram models with order n from 1–4.  
  - **Jelinek–Mercer Smoothing**: Jelinek–Mercer interpolation where we tune λ via perplexity minimization.  
- **Speech rate (SR)**: Units per second, computed from semantically similar texts using the formula SR = NU/D where NU is the number of units in a specific semantically similar text and D is the duration it took to read that text (phonotation time, excluding pauses). 
- **Information rate (IR)**: The rate at which inforamtion is transmitted (bit/s) computed by IR = ID * SR
- To get an IR estimate for a specific language, we average over all speakers from that language 
 
---

## 📂 Data  

### Sources  
- **Written data**: [Tatoeba subtitles corpus (OPUS, v2023-04-12)](https://opus.nlpl.eu/) (tokenized).  
- **Spoken data**: 15 [semantically similar texts](https://github.com/ETHZMSProjects/Coupe_Expansion/tree/master/emillys_code/semantically_similar_texts) from Coupé et al. (2019).
- Measures of time it took each speaker to read each of the semantically similar texts [("phonotation time")](https://github.com/ETHZMSProjects/Coupe_Expansion/blob/master/AutomaticSylDetect.csv) from Coupé et al. (2019)

---

## Supported Languages

| Language          | Code | 
|-------------------|------|
| French            | FRA  |
| German            | DEU  | 
| English           | ENG  |

---

This repository supports two workflows:

1. **Interactive Data Preperation Pipeline**
2. **Command Line Pipeline for heavy IR Computation**
   
### 1) Interactive Data Preperation Pipeline
Allows to run the preprocessing steps, verifying all data exists and is complete (using corpus_size = 'max')
Besides, this pipeline can be used for sanity checks and debugging using a tiny dataset (e.g. using corpus_size = 100)
- Ensures folder structure under <folder_name>/<LANG>/[phones|sylls]/.
- Calls check_ready_to_run(...) and check_data_availability(...) which verifies the existence and completeness of required pickle and csv files:
  -  Verifies expected pickles under <folder>/<language>/[phones|sylls]/ with largest possible corpus size 
  -  Verifies counts CSV at semantically_similar_texts/ling_units_counts.csv with required columns (phones → {"language","n_phones"}, sylls → {"language","n_syllables"})
- If missing/incomplete, you’ll be prompted to generate via parse_to_phones_and_sylls(...) or count_ling_units(...).
- Writes a manifest and .ok file which indicates that all necessary data is complete.
- Produces a small summary table via create_minimal_summary(...) for sanity checks
  
**Interactive prompts: This pipeline is not intended for non-interactive or heavy runs (cluster jobs, CI).**

#### Preprocessing  
1. Converts each word into International Phonetic Alphabet (**IPA**) via [espeak-ng](https://github.com/espeak-ng/espeak-ng) and [CharsiuG2P](https://arxiv.org/abs/2204.03067).
2. Normalizes each word into Unicode NFC, converts digit to words and merges clitics.
3. Cleaning: removes suprasegmentals, punctuation and cartifacts.
4. Syllabification: rule-based **Maximal Onset Principle** with language-specific, data-driven legal onsets.
5. Counts the number of each linguistic unit type in the semantically similar texts

---

### 2) Command Line Pipeline for heavy IR Computation
For each (language, unit_type, text_type) task:
- Checks manifest which indicates the combination of language and unit_type is ready (preprocessing has been completed)
- Can run preprocessing via parse_to_phones_and_sylls(...) if needed
- For each n in --n-values, train Jelinek–Mercer smoothed n-gram models.
  - Computes lambdas tuned on dev perplexity; logs show best λ, mean IR, and perplexity scores.
  - Metrics per n: ID, IR, SR, dev_perplexity, test_perplexity.
- If using largest possible corpus (corpus_size = 'max), automatically updates Metrics CSV via update_values_in_csv(...) and saves model files via jm_model.save_model(...).

#### File Structure & Naming
```
produced_data_large_corpus/<LANG>/
  phones/phonized_<LANG>_size:*.pkl
  sylls/syllabified_<LANG>_size:*.pkl
  ipa_corpus_<LANG>_size:*.pkl     # words live directly under <LANG>/
```
---

## 🖥 Installation & Dependencies  

- **Python ≥ 3.10**  
- Dependencies: `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`  
- External tools: [espeak-ng](https://github.com/espeak-ng/espeak-ng) (needs to be installed as subprocess, indicating the storage path in the code), [CharsiuG2P](https://arxiv.org/abs/2204.03067), [num2words](https://github.com/savoirfairelinux/num2words).  


git clone https://github.com/ETHZMSProjects/Coupe_Expansion.git
cd Coupe_Expansion/emillys_code
pip install -r requirements.txt
---

#### 🚀 Usage

1. Complete installation of tools and dependencies
2. Download written corpus (see Sources above)
3. Get phonotation time measures (see Sources above)
4. Specify the path to the written Corpus under "Sentence Data": "../../data/CMN/cmn.tok" in language_config.json
5. Adjust corpus_size in language_config.json if needed
6. Run run_test_pipeline() in data_prep.ipynb to prepare all required data
7. For heavy runs, use IR_comp_pipeline.py:

IR_comp_pipeline.py accepts language/unit_type/text type settings, n-gram orders, corpus size, number of parallel workers, logging, and RAM budgeting.

In terminal, set your parameters and run:
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

Notes & constraints:
- Valid --unit-types: phones, sylls, words
- Valid --text-types: across_sentences, within_words
- Impossible combo: words + within_words is excluded automatically.
- corpus-size: "max" or an integer (e.g., 100 for testing).
- IR computation takes more than 24h for syllables and words with higher n values
- Parallelism is capped using RAM heuristics (see resource_manager.py)
---

## 📈 Visualizations

Run `plotting.ipynb` to generate plots for:

- Information Rate Comparison across Languages  
- Information Rate vs. N-gram Order

---

## 📖 References

Coupé, C., Oh, Y. M., Dediu, D., & Pellegrino, F. (2019). Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche. Science advances, 5(9), eaaw2594. https://doi.org/10.1126/sciadv.aaw2594 <br/>

Pellegrino, F., Coupé, C., & Marsico, E. (2011). Across-Language Perspective on Speech Information Rate. Language, 87, 539 - 558. <br/>

Oh, Y.M. (2015). Linguistic complexity and information : quantitative approaches. 

---
## 👤 Author
**Emilly Sidaine-Daumiller**  
Master’s student in Neural Systems and Computation at University and ETH Zurich 

---
                

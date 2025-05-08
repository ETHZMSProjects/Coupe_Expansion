# Human Speech Information Rate across Languages <br/> Expansion of Coupé et al. (2019)

This project estimates **information density** and **information rate** from syllabified linguistic data using **n-gram Markov models**. It supports multiple languages and allows comparing information rate and information density values for each language **across different ngram orders and linguistic unit types** (syllables, characters and phonemes).

---

## Theoretical Background

This project builds on the methodology from [Coupé et al. (2019)](https://www.science.org/doi/10.1126/sciadv.aaw2594), which used **first and second-order Markov models** over within word syllable transitions to compute **information density** and **information rate** (bits/sec) for each language. To estimate the speech rate, they collected data from 170 native speakers across 17 languages who read a semantically similar text out loud. From this data they derive what they claim is an average information rate
across languages that is around 39 bits/s.


##  Table of Contents

- [Repository Structure](#-repository-structure)  
- [ Supported Languages](#-supported-languages)  
- [ Processing Types](#-processing-types)  
- [ Desciption of the Pipeline](#-description-of-the-pipeline)
- [ Computed Metrics](#-computed-metrics)  
- [🚀 Running the Pipeline](#-running-the-pipeline)  
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
| Mandarin Chinese  | CMN  | 🇨🇳   |
| Cantonese         | YUE  | 🇭🇰   |
| Japanese          | JPN  | 🇯🇵   |
| Vietnamese        | VIE  | 🇻🇳   |

---

## Processing Types

The models currently support datasets consisting of phonetically transcribed words in IPA, split into syllables.

 _Planned extensions_: character-level, phoneme-level or extending to sentence-level data.

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

## Computed Metrics

This project computes:  <br/> 
<img src="emillys_code/images/formulas.png" alt="Formulas" width="400"/>

---
## Running the Pipeline

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

## Visualizations

Run `plotting.ipynb` to generate plots for:

- Information Rate Comparison across Languages  
- Information Rate vs. N-gram Order

## References

Coupé, C., Oh, Y. M., Dediu, D., & Pellegrino, F. (2019). Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche. Science advances, 5(9), eaaw2594. https://doi.org/10.1126/sciadv.aaw2594 <br/>

Pellegrino, F., Coupé, C., & Marsico, E. (2011). Across-Language Perspective on Speech Information Rate. Language, 87, 539 - 558. <br/>

Oh, Y.M. (2015). Linguistic complexity and information : quantitative approaches. 

---
## 👤 Author
**Emilly Sidaine-Daumiller**  
Master’s student in Neural Systems and Computation at University and ETH Zurich 

---
                

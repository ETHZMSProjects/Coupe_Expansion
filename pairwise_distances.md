# Pairwise Distance Analysis - Usage Guide

## Overview

This script performs pairwise distance analysis between languages based on three linguistic measures, following the methodology from "Different languages, similar encoding efficiency" study. The script is designed to be flexible, allowing you to analyze different linguistic units (syllables, words, phonemes, etc.) and their corresponding rates and information densities.

## Prerequisites

### Software Requirements
1. **R**: Download and install R from [https://cran.r-project.org/](https://cran.r-project.org/)
2. **RStudio** (recommended): Download and install from [https://posit.co/download/rstudio-desktop/](https://posit.co/download/rstudio-desktop/)

### Required R Packages
The script will automatically install missing packages, but you can install them manually:

```r
install.packages(c("philentropy", "reshape2", "broman", "ggplot2", "dplyr", "knitr"))
```

### Data Requirements
Your CSV file must contain:
- **Language column**: Language identifiers (e.g., "ENG", "FRA", "DEU")
- **Unit count column**: Number of linguistic units (e.g., syllables, words)
- **Duration column**: Time duration of utterances
- **Information density column**: Information content per unit

Optional:
- **Family column**: Language family information (will be auto-generated for common languages if missing)

### Data Format Notes
⚠️ **Important**: The script expects tab-separated files by default. If your CSV uses comma separators, you need to modify this line in the script:

```r
# Change this line:
info.rate.data <- read.table(input_file, header=TRUE, sep="\t", quote="", stringsAsFactors=FALSE)

# To this:
info.rate.data <- read.table(input_file, header=TRUE, sep=",", quote="", stringsAsFactors=FALSE)
```

You can check your file format by opening it in a text editor - look for tabs (`\t`) or commas (`,`) between values.

## Configuration

### Step 1: Set Your Column Names
At the top of the script, modify these variables to match your data:

```r
# Column names in your CSV file:
unit_count_column <- "NS"        # Your unit count column name
duration_column <- "Duration"    # Your duration column name  
information_density_column <- "ID"  # Your information density column name
```

### Step 2: Set Display Labels
These labels will appear in plots and results:

```r
# Output labels for your measures:
unit_count_label <- "NS"         # How you want unit count displayed
unit_rate_label <- "SR"          # How you want unit rate displayed
info_rate_label <- "IR"          # How you want information rate displayed
```

### Step 3: Set Experiment Details
**IMPORTANT**: Change the experiment name for each different analysis:

```r
# Experiment name (used for output folder and file names):
experiment_name <- "syllable_analysis"  # CHANGE THIS FOR EACH EXPERIMENT!

# Input file path:
input_file <- "./InfoRateData.csv"  # Path to your data file
```

## Example Configurations

### Example 1: Syllable-Based Analysis (Default)
```r
unit_count_column <- "NS"
duration_column <- "Duration"
information_density_column <- "ID"
unit_count_label <- "NS"
unit_rate_label <- "SR"
info_rate_label <- "IR"
experiment_name <- "syllable_analysis"
```

### Example 2: Word-Based Analysis
```r
unit_count_column <- "NW"
duration_column <- "Duration"
information_density_column <- "WID"
unit_count_label <- "NW"
unit_rate_label <- "WR"
info_rate_label <- "WIR"
experiment_name <- "word_analysis"
```

### Example 3: Second-Order Markov Analysis
```r
unit_count_column <- "NS"
duration_column <- "Duration"
information_density_column <- "ID_2nd_order"
unit_count_label <- "NS"
unit_rate_label <- "SR"
info_rate_label <- "IR_2nd"
experiment_name <- "second_order_markov"
```

### Example 4: Phoneme-Based Analysis
```r
unit_count_column <- "NP"
duration_column <- "Duration"
information_density_column <- "PID"
unit_count_label <- "NP"
unit_rate_label <- "PR"
info_rate_label <- "PIR"
experiment_name <- "phoneme_analysis"
```

## Safety Features

### Automatic Folder Creation
The script automatically creates timestamped output folders to prevent overwrites:
- Format: `results_[experiment_name]_[timestamp]`
- Example: `results_syllable_analysis_20241203_143022`

### File Organization
Each run creates a separate folder containing:
- **`analysis_data.csv`**: Your processed data with calculated unit rates and information rates
- **`pairwise_distances.csv`**: Complete matrix of pairwise distances between all language pairs for each measure and distance method
- **`permutation_test_results.csv`**: Statistical test results comparing measures (p-values, means, standard deviations)
- **`summary_statistics.csv`**: Descriptive statistics (mean, median, SD, min, max) for each measure and distance method
- **`boxplots.png`**: Box and whisker plot visualization showing distribution of distances
- **`analysis_report.txt`**: Human-readable summary report with configuration details and key findings

### Final Output Structure
After running the analysis, you'll find a new folder with this structure:
```
results_[experiment_name]_[timestamp]/
├── analysis_data.csv          # Processed input data
├── pairwise_distances.csv     # All distance calculations  
├── permutation_test_results.csv # Statistical comparisons
├── summary_statistics.csv     # Descriptive statistics
├── boxplots.png              # Visualization
└── analysis_report.txt       # Summary report
```

Example folder name: `results_syllable_analysis_20241203_143022`

## Running the Analysis

### Step 1: Prepare Your Data
1. Ensure your CSV file has the required columns
2. Check that there are no missing values in critical columns
3. Verify language codes are consistent

### Step 2: Configure the Script
1. Open the R script
2. Modify the configuration variables at the top
3. **Double-check the experiment name** - this prevents accidental overwrites

### Step 3: Execute the Script
1. Set your working directory to where your CSV file is located
2. Run the entire script
3. Monitor the console output for progress and any warnings

### Step 4: Review Results
1. Check the generated output folder
2. Review the `analysis_report.txt` for a summary
3. Examine the statistical results and visualizations

## Understanding the Output

### Box Plots
The script generates box and whisker plots showing the distribution of pairwise distances for each measure across five different distance methods:
- **K-S**: Kolmogorov-Smirnov test
- **K-L**: Kullback-Leibler divergence
- **J-S**: Jensen-Shannon divergence
- **H**: Hellinger distance
- **Chi²**: Squared Chi distance

### Statistical Tests
Paired permutation t-tests compare the three measures:
- Tests whether languages are significantly more similar in one measure vs. another
- Uses 1,000 permutations for robust statistical inference
- P-values < 0.05 indicate significant differences

### Key Research Question
The analysis tests whether languages converge on similar **information transmission rates** despite varying in their structural properties (unit counts and unit rates).

## Common Use Cases

### 1. Comparing Information Density Calculations
Use different methods to compute information density (1st order, 2nd order Markov, etc.) while keeping the same linguistic units:

```r
# Run 1: First-order
information_density_column <- "ID_1st"
info_rate_label <- "IR_1st"
experiment_name <- "first_order_analysis"

# Run 2: Second-order  
information_density_column <- "ID_2nd"
info_rate_label <- "IR_2nd"
experiment_name <- "second_order_analysis"
```

### 2. Comparing Linguistic Units
Test different segmentation approaches:

```r
# Run 1: Syllables
unit_count_column <- "NS"
unit_rate_label <- "SR"
experiment_name <- "syllable_units"

# Run 2: Words
unit_count_column <- "NW"
unit_rate_label <- "WR"
experiment_name <- "word_units"
```

### 3. Cross-Linguistic Comparisons
Compare results across different language samples by using different input files:

```r
# Run 1: European languages
input_file <- "./european_languages.csv"
experiment_name <- "european_sample"

# Run 2: All languages
input_file <- "./all_languages.csv"
experiment_name <- "full_sample"
```

## Troubleshooting

### Common Errors

**"Missing required columns"**
- Check that your column names match exactly (case-sensitive)
- Verify your CSV file has headers

**"Input file not found"**
- Check the file path in `input_file`
- Ensure the file exists in the specified location

**"Cannot compute pairwise distances"**
- Check for missing values (NAs) in your data
- Ensure all values are numeric where expected

**File loading issues**
- If you get parsing errors, check whether your file uses commas or tabs as separators
- Modify the `sep=` parameter in the `read.table()` function accordingly
- Common separators: `sep="\t"` (tabs), `sep=","` (commas), `sep=";"` (semicolons)

### Performance Notes
- The permutation tests can take several minutes to complete
- Larger datasets will take longer to process
- Consider reducing permutation count (n.perm) for faster testing

## Best Practices

### 1. Version Control
- Save copies of your configured script for each experiment
- Document your parameter choices
- Keep track of which results correspond to which configurations

### 2. Data Validation
- Always examine your input data first
- Check for outliers or unusual values
- Verify that calculated rates make linguistic sense

### 3. Result Interpretation
- Compare results across different configurations
- Look for consistent patterns across distance methods
- Consider the linguistic significance of your findings

### 4. Documentation
- Keep notes on your experimental design
- Document any data preprocessing steps
- Record your interpretation of results

## Advanced Usage

### Modifying Distance Methods
You can add or remove distance methods by modifying the `methods` vector:

```r
methods <- c("Kolmogorov–Smirnov", "Kullback-Leibler", "Jensen-Shannon", "Hellinger", "Squared-Chi")
```

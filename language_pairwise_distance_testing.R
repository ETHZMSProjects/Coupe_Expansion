# Flexible Pairwise Distance Analysis Script
# Adapted from "Different languages, similar encoding efficiency" study
# Allows students to analyze different linguistic measures

# ========================================
# CONFIGURATION - CHANGE THESE VARIABLES
# ========================================

# Set your column names here - these are the three measures you want to analyze
# Default values correspond to the original study:
# - NS: Number of Syllables
# - SR: Speech Rate (calculated as NS/Duration)  
# - IR: Information Rate (calculated as SR * ID)

# Column names in your CSV file:
unit_count_column <- "NS"        # e.g., "NS" for syllables, "NW" for words
duration_column <- "Duration"    # Duration column (should remain consistent)
information_density_column <- "ID"  # Information density column (e.g., "ID" for syllables, "WID" for words)

# Output labels for your measures (used in plots and results):
unit_count_label <- "NS"         # e.g., "NS", "NW" 
unit_rate_label <- "SR"          # e.g., "SR" for syllable rate, "WR" for word rate
info_rate_label <- "IR"          # e.g., "IR", "WIR" for word information rate

# Experiment name (used for output folder and file names):
experiment_name <- "syllable_analysis"  # Change this for each experiment!

# Input file path:
input_file <- "./InfoRateData.csv"  # Path to your CSV file

# ========================================
# AUTOMATIC FOLDER CREATION AND SETUP
# ========================================

# Create output directory with timestamp to avoid overwrites
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_dir <- paste0("./results_", experiment_name, "_", timestamp)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("Output directory created:", output_dir, "\n")
cat("All results will be saved in this folder.\n\n")

# ========================================
# LOAD REQUIRED LIBRARIES
# ========================================

# Check and install required packages
required_packages <- c("philentropy", "reshape2", "broman", "ggplot2", "dplyr", "knitr")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if(length(missing_packages)) {
  cat("Installing missing packages:", paste(missing_packages, collapse=", "), "\n")
  install.packages(missing_packages)
}

# Load libraries
suppressMessages({
  library(philentropy)
  library(reshape2)
  library(broman)
  library(ggplot2)
  library(dplyr)
  library(knitr)
})

# ========================================
# DATA LOADING AND PREPARATION
# ========================================

cat("Loading data from:", input_file, "\n")

# Load the data
if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file, "\nPlease check the file path.")
}

info.rate.data <- read.table(input_file, header=TRUE, sep=",", quote="", stringsAsFactors=FALSE)

# Verify required columns exist
required_cols <- c(unit_count_column, duration_column, information_density_column, "Language")
missing_cols <- required_cols[!(required_cols %in% names(info.rate.data))]

if(length(missing_cols)) {
  stop("Missing required columns in data: ", paste(missing_cols, collapse=", "))
}

cat("Data loaded successfully. Columns found:", paste(names(info.rate.data), collapse=", "), "\n")

# Compute derived measures
cat("Computing derived measures...\n")

# Unit Rate (e.g., Speech Rate for syllables, Word Rate for words)
info.rate.data$UNIT_RATE <- info.rate.data[,unit_count_column] / info.rate.data[,duration_column]

# Information Rate (Unit Rate * Information Density)
info.rate.data$INFO_RATE <- info.rate.data$UNIT_RATE * info.rate.data[,information_density_column]

# Create analysis dataframe with standardized column names
analysis_data <- data.frame(
  Language = info.rate.data$Language,
  UNIT_COUNT = info.rate.data[,unit_count_column],
  UNIT_RATE = info.rate.data$UNIT_RATE,
  INFO_RATE = info.rate.data$INFO_RATE,
  stringsAsFactors = FALSE
)

# Add language family data if available
if ("Family" %in% names(info.rate.data)) {
  analysis_data$Family <- info.rate.data$Family
} else {
  # Add family data based on language codes
  analysis_data$Family <- sapply(as.character(analysis_data$Language), function(a) {
    switch(a,
           "CAT" = "Indo-European",
           "CMN" = "Sino-Tibetan", 
           "DEU" = "Indo-European",
           "ENG" = "Indo-European",
           "EUS" = "Basque",
           "FIN" = "Uralic",
           "FRA" = "Indo-European",
           "HUN" = "Uralic",
           "ITA" = "Indo-European",
           "JPN" = "Japanese",
           "KOR" = "Korean",
           "SPA" = "Indo-European",
           "SRP" = "Indo-European",
           "THA" = "Tai-Kadai",
           "TUR" = "Turkic",
           "VIE" = "Austroasiatic",
           "YUE" = "Sino-Tibetan",
           "Unknown")
  })
}

# Convert to factors
analysis_data$Language <- as.factor(analysis_data$Language)
analysis_data$Family <- as.factor(analysis_data$Family)

cat("Analysis data prepared with", nrow(analysis_data), "observations and", length(unique(analysis_data$Language)), "languages.\n\n")

# ========================================
# PAIRWISE DISTANCE COMPUTATION
# ========================================

cat("Computing pairwise distances...\n")

# Function to compute pairwise distances for a variable
.compute.pairwise.distances.for.variable <- function(grouping.var, measured.var, normalize=TRUE) {
  require(philentropy)
  
  # Checks
  if( is.null(grouping.var) || is.null(measured.var) || 
      length(grouping.var) != length(measured.var) || 
      anyNA(grouping.var) || anyNA(measured.var) ) {
    warning("Cannot compute pairwise distances for variables that are NULL, contain NAs or have different lengths\n")
    return (NULL)
  }
  if( is.factor(grouping.var) ) grouping.var <- as.character(grouping.var)
  
  # Get the best histogram breaks for the whole of measured.var
  h.all <- hist(measured.var, plot=FALSE)$breaks
  
  # All ordered pairs of grouping.var values
  grouping.var.pairs <- expand.grid("v1"=unique(grouping.var), "v2"=unique(grouping.var), stringsAsFactors=FALSE)
  grouping.var.pairs <- grouping.var.pairs[ grouping.var.pairs$v1 < grouping.var.pairs$v2, ]
  grouping.var.pairs <- grouping.var.pairs[ order(grouping.var.pairs$v1, grouping.var.pairs$v2), ]
  
  # For each such pair, compute the distances
  d <- do.call(rbind, lapply(1:nrow(grouping.var.pairs), function(i) {
    # Select the corresponding measured variable values
    s1 <- (grouping.var == grouping.var.pairs$v1[i])
    s2 <- (grouping.var == grouping.var.pairs$v2[i])
    
    if( sum(s1) == 0 || sum(s2) == 0 ) {
      return (c("Kolmogorov–Smirnov"=NA,
                "Kullback-Leibler"  =NA,
                "Jensen-Shannon"    =NA,
                "Hellinger"         =NA,
                "Squared-Chi"       =NA))
    }
    
    # The histograms of the values
    h1 <- hist(measured.var[s1], breaks=h.all, plot=FALSE)
    h2 <- hist(measured.var[s2], breaks=h.all, plot=FALSE)
    x1 <- h1$counts
    x2 <- h2$counts
    x1[x1==0] <- 1.0e-20  # remove zeros
    x2[x2==0] <- 1.0e-20
    x1 <- x1 / sum(x1)    # normalize between 0 and 1
    x2 <- x2 / sum(x2)
    h12 <- rbind(x1, x2)  # as matrix (required by philentropy::distance())
    
    if( normalize ) {
      est.prob <- NULL
    } else {
      est.prob <- "empirical"
    }
    
    c("Kolmogorov–Smirnov"=as.numeric(suppressWarnings(ks.test(measured.var[s1], measured.var[s2])$statistic)),
      "Kullback-Leibler"  =as.numeric(philentropy::distance(h12, method="kullback-leibler", unit="log", est.prob=est.prob)),
      "Jensen-Shannon"    =as.numeric(philentropy::distance(h12, method="jensen-shannon",   unit="log", est.prob=est.prob)),
      "Hellinger"         =as.numeric(philentropy::distance(h12, method="hellinger",        unit="log", est.prob=est.prob)),
      "Squared-Chi"       =as.numeric(philentropy::distance(h12, method="squared_chi",      unit="log", est.prob=est.prob)))
  }))
  
  grouping.var.pairs <- cbind(grouping.var.pairs, d)
  return (grouping.var.pairs)
}

# Compute pairwise language distances for the three measures
measures <- c("UNIT_COUNT", "UNIT_RATE", "INFO_RATE")
measure_labels <- c(unit_count_label, unit_rate_label, info_rate_label)

tmp <- capture.output({
  pairwise.dists <- do.call(rbind, lapply(1:3, function(i) {
    s <- measures[i]
    label <- measure_labels[i]
    cat("Computing distances for", label, "...\n")
    tmp <- .compute.pairwise.distances.for.variable(analysis_data$Language, analysis_data[,s])
    return (cbind("Measure"=label, tmp))
  }))
}, type="message")

# Clean up column names
names(pairwise.dists)[1+(1:2)] <- paste0("Language",1:2)

# Add family information
pairwise.dists <- merge(pairwise.dists, unique(analysis_data[,c("Language", "Family")]), 
                        by.x="Language1", by.y="Language", all.x=TRUE)
names(pairwise.dists)[ncol(pairwise.dists)] <- "LgFam1"

pairwise.dists <- merge(pairwise.dists, unique(analysis_data[,c("Language", "Family")]), 
                        by.x="Language2", by.y="Language", all.x=TRUE)
names(pairwise.dists)[ncol(pairwise.dists)] <- "LgFam2"

# Reorder columns
pairwise.dists <- pairwise.dists[,c("Language1", "Language2", "LgFam1", "LgFam2", "Measure", 
                                    "Kolmogorov–Smirnov", "Kullback-Leibler", "Jensen-Shannon", 
                                    "Hellinger", "Squared-Chi")]
pairwise.dists <- pairwise.dists[order(pairwise.dists$Language1, pairwise.dists$Language2, pairwise.dists$Measure), ]

# Convert to long format for plotting
pairwise.dists.long <- melt(pairwise.dists, 
                            measure.vars=c("Kolmogorov–Smirnov", "Kullback-Leibler", "Jensen-Shannon", 
                                           "Hellinger", "Squared-Chi"), 
                            variable.name="Method", value.name="Distance")

pairwise.dists.long$Measure <- factor(pairwise.dists.long$Measure, levels = measure_labels)

cat("Pairwise distances computed successfully.\n\n")

# ========================================
# VISUALIZATION: BOX AND WHISKER PLOTS
# ========================================

cat("Creating box and whisker plots...\n")

# Create the box and whisker plots
d <- pairwise.dists.long
plot_list <- list()

# Create individual plots for each distance method
methods <- c("Kolmogorov–Smirnov", "Kullback-Leibler", "Jensen-Shannon", "Hellinger", "Squared-Chi")
method_labels <- c("K-S", "K-L", "J-S", "H", "Chi²")

for(i in 1:length(methods)) {
  method <- methods[i]
  label <- method_labels[i]
  
  plot_list[[i]] <- ggplot(d[d$Method == method,], aes(x=Measure, y=Distance, color=Measure, fill=Measure)) + 
    geom_boxplot(alpha=0.3) + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1), 
          legend.position="none", 
          axis.title.x=element_blank(), 
          axis.title.y=element_blank()) + 
    ggtitle(label) +
    scale_color_manual(values=c("#F8766D", "#00BA38", "#619CFF")) +
    scale_fill_manual(values=c("#F8766D", "#00BA38", "#619CFF"))
}

# Function to arrange multiple plots
multiplot <- function(..., plotlist=NULL, cols=1) {
  library(grid)
  plots <- c(list(...), plotlist)
  numPlots <- length(plots)
  
  if (numPlots==1) {
    print(plots[[1]])
  } else {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(ceiling(numPlots/cols), cols)))
    for (i in 1:numPlots) {
      row_pos <- ceiling(i/cols)
      col_pos <- ((i-1) %% cols) + 1
      print(plots[[i]], vp = viewport(layout.pos.row = row_pos, layout.pos.col = col_pos))
    }
  }
}

# Save plots
png(file.path(output_dir, "boxplots.png"), width=1200, height=300, res=150)
multiplot(plotlist=plot_list, cols=5)
dev.off()

# Display the plots
multiplot(plotlist=plot_list, cols=5)

cat("Box plots saved to:", file.path(output_dir, "boxplots.png"), "\n")

# ========================================
# PAIRED PERMUTATION T-TESTS
# ========================================

cat("Performing paired permutation t-tests...\n")

# Set up comparisons between measures
pairs.of.measures.t.tests <- expand.grid("m1"=as.character(unique(pairwise.dists.long$Measure)), 
                                         "m2"=as.character(unique(pairwise.dists.long$Measure)), 
                                         "d"=as.character(unique(pairwise.dists.long$Method)), 
                                         stringsAsFactors=FALSE)
pairs.of.measures.t.tests <- pairs.of.measures.t.tests[pairs.of.measures.t.tests$m1 < pairs.of.measures.t.tests$m2, ]
pairs.of.measures.t.tests <- pairs.of.measures.t.tests[order(pairs.of.measures.t.tests$m1, pairs.of.measures.t.tests$m2, pairs.of.measures.t.tests$d), ]

# Perform paired permutation t-tests
cat("This may take a few minutes...\n")
pairs.of.measures.t.tests <- cbind(pairs.of.measures.t.tests, 
                                   do.call(rbind, lapply(1:nrow(pairs.of.measures.t.tests), function(i) {
                                     s1 <- (as.character(pairwise.dists.long$Measure) == as.character(pairs.of.measures.t.tests$m1[i]) & 
                                              as.character(pairwise.dists.long$Method) == as.character(pairs.of.measures.t.tests$d[i]))
                                     s2 <- (as.character(pairwise.dists.long$Measure) == as.character(pairs.of.measures.t.tests$m2[i]) & 
                                              as.character(pairwise.dists.long$Method) == as.character(pairs.of.measures.t.tests$d[i]))
                                     
                                     diff_values <- pairwise.dists.long$Distance[s1] - pairwise.dists.long$Distance[s2]
                                     
                                     return (c("mean1"=mean(pairwise.dists.long$Distance[s1], na.rm=TRUE), 
                                               "median1"=median(pairwise.dists.long$Distance[s1], na.rm=TRUE), 
                                               "sd1"=sd(pairwise.dists.long$Distance[s1], na.rm=TRUE), 
                                               "mean2"=mean(pairwise.dists.long$Distance[s2], na.rm=TRUE), 
                                               "median2"=median(pairwise.dists.long$Distance[s2], na.rm=TRUE), 
                                               "sd2"=sd(pairwise.dists.long$Distance[s2], na.rm=TRUE),
                                               "p"=paired.perm.test(diff_values, n.perm=1000)))
                                   })))

# ========================================
# RESULTS SUMMARY AND SAVING
# ========================================

cat("Generating results summary...\n")

# Summary statistics
summary_stats <- pairwise.dists.long %>%
  group_by(Measure, Method) %>%
  summarise(
    mean = round(mean(Distance, na.rm=TRUE), 4),
    median = round(median(Distance, na.rm=TRUE), 4),
    sd = round(sd(Distance, na.rm=TRUE), 4),
    min = round(min(Distance, na.rm=TRUE), 4),
    max = round(max(Distance, na.rm=TRUE), 4),
    .groups = 'drop'
  )

# Save all results to files
write.csv(analysis_data, file.path(output_dir, "analysis_data.csv"), row.names=FALSE)
write.csv(pairwise.dists, file.path(output_dir, "pairwise_distances.csv"), row.names=FALSE)
write.csv(pairs.of.measures.t.tests, file.path(output_dir, "permutation_test_results.csv"), row.names=FALSE)
write.csv(summary_stats, file.path(output_dir, "summary_statistics.csv"), row.names=FALSE)

# Create a detailed results report
report_file <- file.path(output_dir, "analysis_report.txt")
cat("# Pairwise Distance Analysis Report\n", file=report_file)
cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n", file=report_file, append=TRUE)
cat("Experiment:", experiment_name, "\n", file=report_file, append=TRUE)
cat("Input file:", input_file, "\n\n", file=report_file, append=TRUE)

cat("## Configuration\n", file=report_file, append=TRUE)
cat("Unit count column:", unit_count_column, "(", unit_count_label, ")\n", file=report_file, append=TRUE)
cat("Duration column:", duration_column, "\n", file=report_file, append=TRUE)
cat("Information density column:", information_density_column, "\n", file=report_file, append=TRUE)
cat("Unit rate label:", unit_rate_label, "\n", file=report_file, append=TRUE)
cat("Information rate label:", info_rate_label, "\n\n", file=report_file, append=TRUE)

cat("## Data Summary\n", file=report_file, append=TRUE)
cat("Languages analyzed:", length(unique(analysis_data$Language)), "\n", file=report_file, append=TRUE)
cat("Language families:", length(unique(analysis_data$Family)), "\n", file=report_file, append=TRUE)
cat("Total observations:", nrow(analysis_data), "\n\n", file=report_file, append=TRUE)

# Display results
cat("\n" %+% "="*60 %+% "\n")
cat("ANALYSIS COMPLETE\n")


cat("Configuration used:\n")
cat("- Unit count:", unit_count_column, "->", unit_count_label, "\n")
cat("- Unit rate:", unit_rate_label, "(calculated as", unit_count_label, "/", duration_column, ")\n")
cat("- Information rate:", info_rate_label, "(calculated as", unit_rate_label, "*", information_density_column, ")\n\n")

cat("Summary Statistics:\n")
print(summary_stats)

cat("\nPermutation Test Results:\n")
print(kable(pairs.of.measures.t.tests, row.names=FALSE, digits=4,
            caption="Paired permutation t-tests comparing measures with 1,000 permutations"))

cat("\nFiles saved in directory:", output_dir, "\n")
cat("- analysis_data.csv: Processed data used for analysis\n")
cat("- pairwise_distances.csv: All pairwise distance calculations\n")
cat("- permutation_test_results.csv: Statistical test results\n")
cat("- summary_statistics.csv: Descriptive statistics by measure and method\n")
cat("- boxplots.png: Box and whisker plots\n")
cat("- analysis_report.txt: Detailed analysis report\n")

cat("\nAnalysis completed successfully!\n")
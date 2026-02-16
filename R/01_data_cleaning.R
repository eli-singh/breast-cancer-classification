# R/01_data_cleaning.R
# Purpose: load wbca data, do train/test split, save processed files
# Run from project root. Outputs -> data/processed/train.csv and test.csv

# 0. ---- setup ----
library(faraway)
library(ggplot2)
library(dplyr)
library(gtsummary)
library(tidyr)
library(car)
library(psych)
library(caret)
library(corrr)
library(glmnet)
library(kableExtra)
library(pROC)

# 1. ---- ensure directories exist ----
dirs <- c("data/raw", "data/processed", "outputs/figures", "outputs/tables", "R", "report")
for (d in dirs) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

# 2. ---- constants ----
seed <- 60903            # reproducible seed
build_percent <- 0.70    # train proportion
prob_threshold <- 0.5    # default classification threshold (used later)

# 3. ---- load data ----
# wbca is in faraway package
data("wbca", package = "faraway")
breast <- wbca

# rename columns and type conversions
colnames(breast) <- c("Class", "Marginal_Adhesion", "Bare_Nuclei",
                      "Bland_Chromatin", "Epithelial_Cell_Size", "Mitoses",
                      "Normal_Nucleoli", "Clump_Thickness",
                      "Cell_Shape_Uniformity", "Cell_Size_Uniformity")

# Confirm class coding and make factor with meaningful levels
# Original dataset uses 0 = malignant, 1 = benign (double-check)
table(breast$Class)
# convert to factor with labels
breast$Class <- factor(breast$Class, levels = c(0, 1), labels = c("Malignant", "Benign"))

# 4. ---- quick checks ----
stopifnot(sum(is.na(breast)) == 0)  # will throw an error if NAs appear
str(breast)
summary(breast)

# 5. ---- train/test split ----
set.seed(seed)
split <- createDataPartition(breast$Class, p = build_percent, list = FALSE)
train <- breast[split, , drop = FALSE]
test  <- breast[-split, , drop = FALSE]

# sanity checks
cat("\nTrain / Test sizes:\n")
cat("Train:", nrow(train), "\nTest:", nrow(test), "\n")
cat("Train class proportions:\n"); print(prop.table(table(train$Class)))
cat("Test class proportions:\n");  print(prop.table(table(test$Class)))

# 6. ---- save processed datasets ----
write.csv(train, file = "data/processed/train.csv", row.names = FALSE)
write.csv(test,  file = "data/processed/test.csv",  row.names = FALSE)

# 7. ---- save a small README about these files ----
readme_text <- c(
  "train.csv: stratified training set (70%) from wbca dataset",
  "test.csv: holdout test set (30%)",
  paste0("created_on: ", Sys.Date()),
  paste0("seed: ", seed),
  paste0("build_percent: ", build_percent)
)
writeLines(readme_text, con = "data/processed/README.txt")

cat("\nSaved processed datasets to data/processed/\n")

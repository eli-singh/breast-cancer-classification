# R/01_data_cleaning.R
# Purpose: load wbca data, do train/test split, save processed files

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
str(breast)
summary(breast)

# 5. ---- train/test split ----
set.seed(seed)
split <- createDataPartition(breast$Class, p = build_percent, list = FALSE)
train <- breast[split, , drop = FALSE]
test  <- breast[-split, , drop = FALSE]

# 6. ---- exploratory data analysis ----
breast_long <- train %>% 
  tidyr::pivot_longer(cols = -c(Class, Class_bin),
                      names_to = "Variable",
                      values_to = "Value")

boxplot_fig <- ggplot(breast_long, aes(x = Class, y = Value)) +
  geom_boxplot() +
  facet_wrap(~ Variable, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "Distribution of Cytological Features by Tumor Classification",
    x = "Tumor Classification",
    y = "Standardized Feature Value"
  )

ggsave("outputs/figures/boxplots_by_class.png",
       boxplot_fig,
       width = 10,
       height = 8,
       dpi = 300)

summary_tbl <- train %>%
  group_by(Class) %>%
  summarise(across(where(is.numeric),
                   list(mean = mean, sd = sd),
                   na.rm = TRUE))

write.csv(summary_tbl,
          "outputs/tables/summary_statistics.csv",
          row.names = FALSE)

cor_matrix <- cor(train %>% dplyr::select(-Class, -Class_bin))

high_cor <- which(abs(cor_matrix) > 0.7 & upper.tri(cor_matrix), arr.ind = TRUE)

cor_tbl <- data.frame(
  Variable_1 = rownames(cor_matrix)[high_cor[,1]],
  Variable_2 = colnames(cor_matrix)[high_cor[,2]],
  Correlation = cor_matrix[high_cor]
)

write.csv(cor_tbl,
          "outputs/tables/high_correlations.csv",
          row.names = FALSE)

# 7. ---- save processed datasets ----
write.csv(train, file = "data/processed/train.csv", row.names = FALSE)
write.csv(test,  file = "data/processed/test.csv",  row.names = FALSE)

# 8. ---- update README about these files ----
readme_text <- c(
  "train.csv: stratified training set (70%) from wbca dataset",
  "test.csv: holdout test set (30%)",
  paste0("created_on: ", Sys.Date()),
  paste0("seed: ", seed),
  paste0("build_percent: ", build_percent)
)
writeLines(readme_text, con = "data/processed/README.txt")
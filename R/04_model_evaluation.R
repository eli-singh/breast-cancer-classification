# R/04_model_evaluation.R
# Purpose: Model evaluation: ROC/AUC, calibration, confusion matrices, etc.

# ---- setup ----
library(dplyr)
library(ggplot2)
library(pROC)
library(caret)
library(tidyr)
library(ggpubr)

dirs <- c("outputs/figures", "outputs/tables")
for (d in dirs) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

# ---- load predictions and true labels ----
glm_preds_path   <- "outputs/tables/glm_test_predictions.csv"
lasso_preds_path <- "outputs/tables/lasso_test_predictions.csv"

if (!file.exists(glm_preds_path) || !file.exists(lasso_preds_path)) {
  stop("Missing prediction files. Run R/02_logistic_model.R and R/03_lasso_model.R first.")
}

glm_df   <- read.csv(glm_preds_path, stringsAsFactors = FALSE)
lasso_df <- read.csv(lasso_preds_path, stringsAsFactors = FALSE)

# Standardize column names

if (!"pred_prob_glm" %in% names(glm_df)) {
  possible <- names(glm_df)[grepl("pred|prob", names(glm_df), ignore.case = TRUE)]
  if (length(possible) >= 1) names(glm_df)[which(names(glm_df) == possible[1])] <- "pred_prob_glm"
}
if (!"pred_prob_lasso_min" %in% names(lasso_df)) {
  possible <- names(lasso_df)[grepl("lasso|min|pred", names(lasso_df), ignore.case = TRUE)]
  if (length(possible) >= 1) names(lasso_df)[which(names(lasso_df) == possible[1])] <- "pred_prob_lasso_min"
}

# Merge into a single df for evaluation
eval_df <- glm_df %>%
  dplyr::select(pred_prob_glm, Class, Class_bin) %>%
  mutate(row_id = row_number()) %>%
  left_join(
    lasso_df %>% dplyr::select(pred_prob_lasso_min, pred_prob_lasso_1se) %>% mutate(row_id = row_number()),
    by = "row_id"
  ) %>%
  dplyr::select(-row_id)

# Ensure label is numeric 0/1 for ROC functions; set positive = "Malignant"
if (!"Class_bin" %in% names(eval_df)) {
  eval_df$Class_bin <- ifelse(as.character(eval_df$Class) == "Malignant", 1, 0)
}

# ---- ROC & AUC ----
roc_glm   <- pROC::roc(response = eval_df$Class_bin, predictor = eval_df$pred_prob_glm, quiet = TRUE)
roc_lasso <- pROC::roc(response = eval_df$Class_bin, predictor = eval_df$pred_prob_lasso_min, quiet = TRUE)

auc_glm   <- pROC::auc(roc_glm)
auc_lasso <- pROC::auc(roc_lasso)

cat(sprintf("AUC (GLM): %.3f\n", as.numeric(auc_glm)))
cat(sprintf("AUC (LASSO): %.3f\n", as.numeric(auc_lasso)))

# Plot ROC curves together
roc_plot <- ggplot() +
  geom_line(aes(x = rev(roc_glm$specificities), y = rev(roc_glm$sensitivities)), linewidth = 1) +
  geom_line(aes(x = rev(roc_lasso$specificities), y = rev(roc_lasso$sensitivities)), linewidth = 1, linetype = "dashed") +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  labs(x = "1 - Specificity", y = "Sensitivity",
       title = "ROC Curve: GLM vs LASSO",
       subtitle = paste0("AUC GLM = ", round(as.numeric(auc_glm), 3),
                         " | AUC LASSO = ", round(as.numeric(auc_lasso), 3))) +
  theme_minimal() +
  annotate("text", x = 0.65, y = 0.15, label = "GLM = solid\nLASSO = dashed", hjust = 0)

ggsave(filename = "outputs/figures/roc_glm_lasso.png", plot = roc_plot, width = 6, height = 5, dpi = 300)

# ---- Precision-Recall curves ----
pred_glm_rocr   <- ROCR::prediction(eval_df$pred_prob_glm, eval_df$Class_bin)
perf_glm_pr     <- ROCR::performance(pred_glm_rocr, "prec", "rec")
pred_lasso_rocr <- ROCR::prediction(eval_df$pred_prob_lasso_min, eval_df$Class_bin)
perf_lasso_pr   <- ROCR::performance(pred_lasso_rocr, "prec", "rec")

pr_glm_df <- data.frame(recall = perf_glm_pr@x.values[[1]], precision = perf_glm_pr@y.values[[1]], model = "GLM")
pr_lasso_df <- data.frame(recall = perf_lasso_pr@x.values[[1]], precision = perf_lasso_pr@y.values[[1]], model = "LASSO")
pr_df <- bind_rows(pr_glm_df, pr_lasso_df)

pr_plot <- ggplot(pr_df, aes(x = recall, y = precision, color = model)) +
  geom_line() +
  labs(title = "Precision-Recall Curve", x = "Recall", y = "Precision") +
  theme_minimal()
ggsave("outputs/figures/pr_curve.png", pr_plot, width = 6, height = 5, dpi = 300)

# ---- Confusion matrices at threshold 0.5 ----
threshold <- 0.5
# For GLM
glm_pred_class <- factor(ifelse(eval_df$pred_prob_glm >= threshold, "Malignant", "Benign"),
                         levels = c("Malignant", "Benign"))
truth <- factor(ifelse(eval_df$Class_bin == 1, "Malignant", "Benign"), levels = c("Malignant", "Benign"))

cm_glm <- caret::confusionMatrix(glm_pred_class, truth, positive = "Malignant")
cm_glm_tab <- as.data.frame(cm_glm$byClass) %>% tibble::rownames_to_column("metric")
write.csv(cm_glm_tab, "outputs/tables/glm_confusion_metrics.csv", row.names = FALSE)
write.csv(as.data.frame(cm_glm$table), "outputs/tables/glm_confusion_matrix.csv", row.names = FALSE)
# For LASSO (lambda.min)
lasso_pred_class <- factor(ifelse(eval_df$pred_prob_lasso_min >= threshold, "Malignant", "Benign"),
                           levels = c("Malignant", "Benign"))
cm_lasso <- caret::confusionMatrix(lasso_pred_class, truth, positive = "Malignant")
cm_lasso_tab <- as.data.frame(cm_lasso$byClass) %>% tibble::rownames_to_column("metric")
write.csv(cm_lasso_tab, "outputs/tables/lasso_confusion_metrics.csv", row.names = FALSE)
write.csv(as.data.frame(cm_lasso$table), "outputs/tables/lasso_confusion_matrix.csv", row.names = FALSE)


# ---- Calibration plots ----
calibration_plot_fn <- function(pred_probs, truth_vec, model_name, n_bins = 10) {
  df <- data.frame(pred = pred_probs, truth = truth_vec)
  df <- df %>% mutate(bin = ntile(pred, n_bins))
  calib <- df %>%
    group_by(bin) %>%
    summarize(mean_pred = mean(pred, na.rm = TRUE),
              obs = mean(truth, na.rm = TRUE),
              n = n()) %>%
    ungroup()
  calib$model <- model_name
  
  p <- ggplot(calib, aes(x = mean_pred, y = obs)) +
    geom_point(size = 2) +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = paste0("Calibration Plot: ", model_name),
         x = "Mean Predicted Probability",
         y = "Observed Proportion (by decile)") +
    theme_minimal()
  return(list(plot = p, table = calib))
}

cal_glm <- calibration_plot_fn(eval_df$pred_prob_glm, eval_df$Class_bin, "GLM")
cal_lasso <- calibration_plot_fn(eval_df$pred_prob_lasso_min, eval_df$Class_bin, "LASSO")

ggsave("outputs/figures/calibration_glm.png", cal_glm$plot, width = 6, height = 5, dpi = 300)
ggsave("outputs/figures/calibration_lasso.png", cal_lasso$plot, width = 6, height = 5, dpi = 300)

write.csv(cal_glm$table, "outputs/tables/calibration_glm.csv", row.names = FALSE)
write.csv(cal_lasso$table, "outputs/tables/calibration_lasso.csv", row.names = FALSE)

# ---- AUC summary ----
auc_tbl <- data.frame(model = c("GLM", "LASSO"),
                      auc = c(as.numeric(auc_glm), as.numeric(auc_lasso)))
write.csv(auc_tbl, "outputs/tables/auc_summary.csv", row.names = FALSE)

summary_tbl <- tibble::tibble(
  model = c("GLM", "LASSO"),
  auc = c(as.numeric(auc_glm), as.numeric(auc_lasso)),
  sens = c(cm_glm$byClass["Sensitivity"], cm_lasso$byClass["Sensitivity"]),
  spec = c(cm_glm$byClass["Specificity"], cm_lasso$byClass["Specificity"])
)
write.csv(summary_tbl, "outputs/tables/evaluation_summary.csv", row.names = FALSE)
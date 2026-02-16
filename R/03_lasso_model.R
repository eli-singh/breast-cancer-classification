# R/03_lasso_model.R
# Purpose: Fit LASSO (penalized logistic regression) using glmnet

# ---- setup ----
library(glmnet)
library(dplyr)
library(broom)

dirs <- c("outputs/figures", "outputs/tables")
for (d in dirs) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

# ---- load data ----
train <- read.csv("data/processed/train.csv", stringsAsFactors = TRUE)
test  <- read.csv("data/processed/test.csv",  stringsAsFactors = TRUE)

train$Class <- as.factor(train$Class)
test$Class  <- as.factor(test$Class)

train$Class_bin <- ifelse(as.character(train$Class) == "Malignant", 1, 0)
test$Class_bin  <- ifelse(as.character(test$Class) == "Malignant", 1, 0)

# ---- Prepare model matrix ----
# Remove Class and Class_bin from predictors
predictor_names <- setdiff(names(train), c("Class", "Class_bin"))

# Build model matrix (automatically handles factors), exclude intercept with -1
x_train <- model.matrix(as.formula(paste0("~ -1 + ", paste(predictor_names, collapse = " + "))),
                        data = train)
y_train <- train$Class_bin

x_test <- model.matrix(as.formula(paste0("~ -1 + ", paste(predictor_names, collapse = " + "))),
                       data = test)
y_test <- test$Class_bin

# ---- Cross-validated LASSO ----
set.seed(60903)
cv_fit <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, nfolds = 10, type.measure = "class")

# save model object
saveRDS(cv_fit, file = "outputs/tables/cv_glmnet_lasso.rds")

# ---- Coefficients at lambda.min and lambda.1se ----
coef_min <- coef(cv_fit, s = "lambda.min")
coef_1se <- coef(cv_fit, s = "lambda.1se")

# convert to dataframe
coef_to_df <- function(coef_sparse) {
  mat <- as.matrix(coef_sparse)
  df <- data.frame(term = rownames(mat), coefficient = as.numeric(mat[,1]), row.names = NULL, stringsAsFactors = FALSE)
  df <- df %>% filter(term != "(Intercept)")
  df <- df %>% arrange(desc(abs(coefficient)))
  return(df)
}

coef_min_df <- coef_to_df(coef_min)
coef_1se_df <- coef_to_df(coef_1se)

write.csv(coef_min_df, "outputs/tables/lasso_coef_lambda_min.csv", row.names = FALSE)
write.csv(coef_1se_df, "outputs/tables/lasso_coef_lambda_1se.csv", row.names = FALSE)

# ---- Predictions on test set ----
test$pred_prob_lasso_min <- predict(cv_fit, newx = x_test, s = "lambda.min", type = "response")[,1]
test$pred_prob_lasso_1se <- predict(cv_fit, newx = x_test, s = "lambda.1se", type = "response")[,1]

# Save predictions
write.csv(test %>% dplyr::select(pred_prob_lasso_min, pred_prob_lasso_1se, Class, Class_bin),
          "outputs/tables/lasso_test_predictions.csv", row.names = FALSE)

# ---- Summary of Results ----
cv_summary <- data.frame(lambda_min = cv_fit$lambda.min,
                         lambda_1se = cv_fit$lambda.1se,
                         cvm_at_min = min(cv_fit$cvm),
                         n_lambda = length(cv_fit$lambda))
write.csv(cv_summary, "outputs/tables/lasso_cv_summary.csv", row.names = FALSE)
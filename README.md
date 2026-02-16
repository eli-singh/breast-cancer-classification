# Breast Cancer Tumor Classification Using Logistic and Penalized Regression

### Author: Eli Singh, M.S. Biostatistics Candidate

### Overview

This project develops and evaluates predictive models to classify breast tumors as malignant or benign using cytological features from the Wisconsin Breast Cancer dataset (n = 681).

Two modeling approaches were compared:

-   Standard Logistic Regression

-   Penalized Logistic Regression (LASSO)

The goal of the project is to evaluate predictive performance, interpret key predictors in clinically meaningful terms, and assess model calibration.

### Data

The dataset contains 681 observations and 9 cytological predictors derived from fine-needle aspiration samples. The outcome variable is tumor classification (Malignant vs Benign).

The dataset was split using stratified sampling:

-   70% training set

-   30% holdout test set

This split preserves the original class proportions.

### Methods

Model development included:

-   Logistic regression (maximum likelihood estimation)

-   Penalized regression using LASSO (10-fold cross-validation)

-   Multicollinearity diagnostics (VIF)

-   Interaction term assessment

-   ROC curve and AUC comparison

-   Confusion matrix evaluation

-   Calibration assessment

### Model Performance

| Model               | AUC    | Sensitivity | Specificity |
|---------------------|--------|-------------|-------------|
| Logistic Regression | 0.9898 | 0.9296      | 0.9545      |
| LASSO Regression    | 0.9904 | 0.9437      | 0.9621      |

Both models demonstrated strong discrimination (AUC \> 0.9). The LASSO model slightly improved sensitivity and specificity; penalization reduced model complexity while preserving predictive performance. Calibration plots show good agreement between predicted probabilities and observed outcomes.

### ROC Curve Comparison

The ROC curves indicate strong discrimination for both models, with the LASSO model having a slightly higher AUC.

### Calibration Assessment

Both models demonstrate reasonable calibration across predicted risk deciles.

### Key Findings

Increased values of cytological abnormalities were associated with significantly higher odds of malignancy. Penalized regression (LASSO) reduced coefficient magnitude and performed slightly better in predictive accuracy. Both models achieved \>90% sensitivity and \>95% specificity in classifying malignant tumors.

### Limitations

Predictors are standardized and unitless, limiting direct clinical interpretation.

External validation was not performed.

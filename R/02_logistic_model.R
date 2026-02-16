# R/02_logistic_model.R
# Fit logistic regression(s), export ORs + CIs, save model object

# ---- setup ----
library(dplyr)
library(broom)
library(car)        
library(kableExtra) 

dirs <- c("outputs/figures", "outputs/tables")
for (d in dirs) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

# ---- load data ----
train <- read.csv("data/processed/train.csv", stringsAsFactors = TRUE)
test  <- read.csv("data/processed/test.csv",  stringsAsFactors = TRUE)

# create numeric binary outcome: 1 = Malignant, 0 = Benign
train$Class_bin <- ifelse(as.character(train$Class) == "Malignant", 1, 0)
test$Class_bin  <- ifelse(as.character(test$Class) == "Malignant", 1, 0)

# ---- Define feature set ----

# exclude Class and Class_bin from predictors
predictor_names <- setdiff(names(train), c("Class", "Class_bin"))
interaction_term <- "Marginal_Adhesion:Clump_Thickness"

# ---- Fit base logistic regression (no interactions) ----
formula_base <- as.formula(paste0("Class_bin ~ ", paste(predictor_names, collapse = " + ")))
glm_base <- glm(formula_base, data = train, family = binomial(link = "logit"))

# Save model object
saveRDS(glm_base, file = "outputs/tables/glm_base_model.rds")

# ---- Diagnostics: VIF (multicollinearity) ----
vif_tbl <- tryCatch({
  vif_vals <- car::vif(glm_base)
  data.frame(variable = names(vif_vals), VIF = as.numeric(vif_vals))
}, error = function(e) {
  data.frame(message = "VIF calculation failed", error = as.character(e))
})
write.csv(vif_tbl, "outputs/tables/glm_vif.csv", row.names = FALSE)

# ---- Coefficients & Odds Ratio ----
coefs <- broom::tidy(glm_base, conf.int = TRUE, exponentiate = FALSE) %>%
  filter(term != "(Intercept)")

# compute odds ratio & CI
or_tbl <- coefs %>%
  mutate(OR = exp(estimate),
         OR_low = exp(conf.low),
         OR_high = exp(conf.high)) %>%
  select(term, estimate, std.error, statistic, p.value, OR, OR_low, OR_high)

write.csv(or_tbl, "outputs/tables/glm_odds_ratios.csv", row.names = FALSE)

# ---- Interaction model ----
# Add interaction only if both variables exist
if (all(c("Marginal_Adhesion", "Clump_Thickness") %in% predictor_names)) {
  formula_int <- as.formula(paste0("Class_bin ~ ",
                                   paste(predictor_names, collapse = " + "),
                                   " + Marginal_Adhesion:Clump_Thickness"))
  glm_int <- glm(formula_int, data = train, family = binomial(link = "logit"))
  saveRDS(glm_int, file = "outputs/tables/glm_interaction_model.rds")
  
  # Tidy interaction coefficients
  int_coefs <- broom::tidy(glm_int, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(OR = exp(estimate), OR_low = exp(conf.low), OR_high = exp(conf.high))
  write.csv(int_coefs, "outputs/tables/glm_interaction_odds_ratios.csv", row.names = FALSE)
} else {
  message("Interaction variables not present; skipping interaction model.")
}

# ---- model summary ----
sink("outputs/tables/glm_base_summary.txt")
print(summary(glm_base))
sink()
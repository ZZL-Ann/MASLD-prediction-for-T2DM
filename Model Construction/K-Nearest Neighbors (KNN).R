# Load required packages
library(kknn)                # Main package for K-Nearest Neighbors modeling
library(dplyr)               # Data manipulation
library(pROC)                # ROC curve plotting and AUC calculation
library(caret)               # Cross-validation and model tuning
library(ggplot2)             # Data visualization
library(tidyr)               # Data reshaping
library(ResourceSelection)   # Hosmer-Lemeshow goodness-of-fit test

# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Load training data from the specified file path
data_test <- read.csv("C:/Users/Desktop/R data/data_test.csv") # Load testing data 

# Selected feature variables
labelVars <- c("Wt","BMI" ,"Lymph","ALB","ALT", "HDL","TG","HbA1c","FCP","Gender","Met","MedStatus","HLP") 

# Create formula for KNN model
formula_knn <- as.formula(paste("Group ~", paste(labelVars, collapse = " + ")))

# Keep only relevant variables in training and testing datasets
trainVars <- c("Group", labelVars)
data_train_clean <- data_train[, trainVars]
data_test_clean  <- data_test[, trainVars]

# Apply log transformation to selected numeric variables to reduce skewness
data_train <- data_train %>%
  mutate(across(all_of(numVars_noboth), ~ log(pmax(., 1e-6)))) 
data_test <- data_test %>%
  mutate(across(all_of(numVars_noboth), ~ log(pmax(., 1e-6))))

# Standardize training set and apply the same parameters to testing set
preproc <- preProcess(data_train[, numVarsAll], method = c("center", "scale"))

data_train_scaled <- data_train
data_train_scaled[, numVarsAll] <- predict(preproc, data_train[, numVarsAll])

data_test_scaled <- data_test
data_test_scaled[, numVarsAll] <- predict(preproc, data_test[, numVarsAll])

# Grid Search Hyperparameter Tuning
# 1. Define hyperparameter ranges
k_values <- seq(10,500,by = 10)   # Number of neighbors
distance_values <- c(1, 2)        # Distance metric parameter
kernel_values <- c("rectangular", "triangular", "epanechnikov", 
                   "biweight", "triweight", "cos", "inv", "gaussian", "optimal") # Kernel functions

# 2. Create 10-fold cross-validation splits
set.seed(123456)
cv_folds <- createFolds(data_train_scaled$Group, k = 10)

# 3. Data frame to store results
results_all <- data.frame()

# 4. Nested loops for grid search
for (k in k_values) {
  for (dist in distance_values) {
    for (ker in kernel_values) {
      
      # Initialize vectors to collect metrics for each fold
      aucs <- c()
      accs <- c()
      recalls <- c()
      precisions <- c()
      f1s <- c()
      
      for (fold_idx in seq_along(cv_folds)) {
        train_idx <- unlist(cv_folds[-fold_idx])
        valid_idx <- unlist(cv_folds[fold_idx])
        
        train_fold <- data_train_scaled[train_idx, ]
        valid_fold <- data_train_scaled[valid_idx, ]
        
        model <- kknn(formula_knn, train = train_fold, test = valid_fold,
                      k = k, distance = dist, kernel = ker)
        
        probs <- model$prob[, "2"]
        preds <- ifelse(probs >= 0.5, "2", "1")  # Binary prediction
        true <- valid_fold$Group
        
        # Calculate AUC
        roc_obj <- roc(true, probs, levels = c("1", "2"), direction = "<")
        aucs <- c(aucs, auc(roc_obj))
        
        # Compute confusion matrix metrics
        cm <- confusionMatrix(factor(preds, levels = c("1", "2")),
                              factor(true, levels = c("1", "2")),
                              positive = "2")
        accs <- c(accs, cm$overall["Accuracy"])
        recalls <- c(recalls, cm$byClass["Recall"])
        precisions <- c(precisions, cm$byClass["Precision"])
        f1s <- c(f1s, cm$byClass["F1"])
      }
      
      # Record average metrics for this hyperparameter combination
      results_all <- rbind(results_all, data.frame(
        k = k, distance = dist, kernel = ker,
        AUC = mean(aucs),
        Accuracy = mean(accs),
        Recall = mean(recalls),
        Precision = mean(precisions),
        F1 = mean(f1s)
      ))
    }}}
# 5. Visualize results to find the hyperparameters that give the highest AUC
results_long <- results_all %>%
  pivot_longer(cols = c("AUC", "Accuracy", "Recall", "Precision", "F1"),
               names_to = "Metric", values_to = "Value")

ggplot(results_long, aes(x = k, y = Value, color = Metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(title = "KNN Performance Metrics vs. k",
       x = "k (Number of Neighbors)", y = "Metric Value") +
  theme_minimal() +
  theme(text = element_text(size = 13))

# Fit final KNN model with optimal hyperparameters
# KNN does not first create a reusable model for predictions. Instead, each time it is invoked, it computes the distance between each sample in the test set and the training set using the training data, thereby making a prediction.
knn_model_train <- kknn(
  formula_knn,
  train = data_train_scaled,
  test = data_train_scaled,   # Predict on training data
  k = 100,
  distance = 1,
  kernel = "gaussian"
)
prob_train <- knn_model_train$prob[, "2"] # Predicted probability for class "2"

# Determine optimal cut-off based on Youden index
roc_train <- roc(data_train_scaled$Group, prob_train, levels = c("1", "2"), direction = "<")
auc(roc_train)
best_threshold_youden <- as.numeric(coords(roc_train, "best", best.method = "youden", ret = "threshold"))
print(best_threshold_youden)

# Apply threshold to training set predictions
pred_train <- ifelse(prob_train >= best_threshold_youden, "2", "1") 
cm_train <- confusionMatrix(factor(pred_train, levels = c("1", "2")),
                            factor(data_train_scaled$Group, levels = c("1", "2")),
                            positive = "2")
cm_train
tp <- cm_train$table["2", "2"]
tn <- cm_train$table["1", "1"]
fp <- cm_train$table["2", "1"]
fn <- cm_train$table["1", "2"]

accuracy <- (tp + tn) / sum(cm_train$table)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
f1 <- 2 * precision * recall / (precision + recall)
npv <- tn / (tn + fn)

# Print training set metrics
cat("Accuracy:", round(accuracy, 6), "\n")
cat("Recall (Sensitivity):", round(recall, 6), "\n")
cat("Precision (PPV):", round(precision, 6), "\n")
cat("NPV:", round(npv, 6), "\n")
cat("F1 Score:", round(f1, 6), "\n")

# Hosmer-Lemeshow test for calibration
observed_train <- ifelse(data_train_scaled$Group == "2", 1, 0)
hoslem_train <- hoslem.test(observed_train, prob_train, g = 10)
print(hoslem_train)

# Brier score for training set
brier_train <- mean((observed_train - prob_train)^2)
cat("Brier Score (Train):", round(brier_train, 6), "\n")

# Compute performance metrics on test set
knn_model <- kknn(
  formula_knn, 
  train = data_train_scaled, 
  test = data_test_scaled,
  k = 100,
  distance = 1,
  kernel = "gaussian"
)
prob_test <- knn_model$prob[, "2"]
pred_test <- ifelse(prob_test >= best_threshold_youden, "2", "1")
cm_test <- confusionMatrix(factor(pred_test, levels = c("1", "2")),
                           factor(data_test_scaled$Group, levels = c("1", "2")),
                           positive = "2")
cm_test

roc_test <- roc(data_test_scaled$Group, prob_test, levels = c("1", "2"), direction = "<")
round(auc(roc_train), 4) # AUROC for test set

tp <- cm_test$table["2", "2"]
tn <- cm_test$table["1", "1"]
fp <- cm_test$table["2", "1"]
fn <- cm_test$table["1", "2"]
accuracy <- (tp + tn) / sum(cm_test$table)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
f1 <- 2 * precision * recall / (precision + recall)
npv <- tn / (tn + fn)

# Print testing set metrics
cat("Accuracy:", round(accuracy, 6), "\n")
cat("Recall (Sensitivity):", round(recall, 6), "\n")
cat("Precision (PPV):", round(precision, 6), "\n")
cat("NPV:", round(npv, 6), "\n")
cat("F1 Score:", round(f1, 6), "\n")

# Hosmer-Lemeshow Test p-value
observed_test <- ifelse(data_test_scaled$Group == "2", 1, 0)
hl_test <- hoslem.test(observed_test, prob_test)
print(hl_test)

# Brier Score for test set
brier_score <- mean((observed_test - prob_test)^2)
cat("Brier Score:", round(brier_score, 6), "\n")
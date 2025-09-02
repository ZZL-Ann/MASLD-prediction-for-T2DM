# Load required packages
library(lightgbm)            # Main package for light gradient boosting machine modeling
library(caret)               # Cross-validation and model tuning
library(pROC)                # ROC curve plotting and AUC calculation
library(ResourceSelection)   # Hosmer-Lemeshow goodness-of-fit test
library(MLmetrics)           # Additional ML metrics statistical tools
library(DescTools)           # Extra statistical tools

# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Load the training dataset from the specified file path
data_test <- read.csv("C:/Users/Desktop/R data/data_test.csv")  # Load the testing dataset 

# Selected feature variables
labelVars <- c("Wt","BMI" ,"Lymph","ALB","ALT", "HDL","TG","HbA1c","FCP","Gender","Met","MedStatus","HLP") 

train_data <- data_train[, labelVars]
test_data <- data_test[, labelVars]

# Obtain the colnames or indices of categorical variables in train_data
categorical_in_model <- intersect(catVars, labelVars)

# Construct lgb.Dataset object for training
dtrain_lgb <- lgb.Dataset(
  data = as.matrix(train_data),
  label = train_label,
  categorical_feature = categorical_in_model
)

# Define the parameter grid for hyperparameter search
param_grid <- expand.grid(
  num_leaves = c(4, 8, 16),
  max_depth = c(2, 3, 4),
  min_data_in_leaf = c(15,20,25,30,35,40),
  learning_rate = seq(0.05, 0.2, by = 0.01),
  lambda_l1 = c(0,0.1,0.5,1),
  lambda_l2 = c(0,0.1,0.5,1),
  bagging_fraction = c(0.6,0.8,0.9),
  feature_fraction = c(0.4,0.5,0.6,0.7,0.8, 0.9)
)

results_all <- data.frame()

# Loop through all parameter combinations
for (i in 1:nrow(param_grid)) {
  set.seed(123)
  # Rebuild Dataset to avoid inheriting old parameters
  dtrain_lgb <- lgb.Dataset(
    data = as.matrix(train_data),
    label = train_label,
    categorical_feature = categorical_in_model
  )
  # Set LightGBM parameters for current combination
  params <- list(
    objective = "binary",
    metric = "auc",
    num_leaves = param_grid$num_leaves[i],
    max_depth = param_grid$max_depth[i],
    min_data_in_leaf = param_grid$min_data_in_leaf[i],
    learning_rate = param_grid$learning_rate[i],
    lambda_l1 = param_grid$lambda_l1[i],
    lambda_l2 = param_grid$lambda_l2[i],
    bagging_fraction = param_grid$bagging_fraction[i],
    feature_fraction = param_grid$feature_fraction[i],
    seed = 2025,
    bagging_freq = 1,
    feature_pre_filter = FALSE,
    verbose = -1
  )
  # Perform 10-fold cross-validation with early stopping
  cv <- lgb.cv(
    params = params,
    data = dtrain_lgb,
    nfold = 10,  
    nrounds = 2000,
    early_stopping_rounds = 50,
    stratified = TRUE,
    verbose = -1
  )
  # Record best iteration and corresponding AUC
  best_iter <- cv$best_iter
  best_score <- max(unlist(cv$record_evals$valid$auc$eval))
  # Store results
  results_all <- rbind(
    results_all,
    cbind(param_grid[i, ], best_iter, best_score)
  )
}

# Sort results by AUC to identify best hyperparameter combination
results_all[order(-results_all$best_score), ][1:10, ]

# Fix best hyperparameters for final model
best_params <- list(
  objective = "binary",
  metric = "auc",
  learning_rate = 0.10,
  num_leaves = 4,
  max_depth = 2,
  min_data_in_leaf = 25,
  lambda_l1 = 1,
  lambda_l2 = 1,
  bagging_fraction = 0.9,
  feature_fraction = 0.9,
  bagging_freq = 1,
  feature_pre_filter = FALSE,
  verbose = -1,
  seed = 2025
)

# Cross-validation to determine best number of iterations
model_final <- lgb.cv(
  params = best_params,
  data = dtrain_lgb,
  nfold = 10,
  nrounds = 1000,
  early_stopping_rounds = 50
)
model_final$best_iter

# Train final LightGBM model with optimal iteration
model_lgb <- lgb.train(
  params = best_params,
  data = dtrain_lgb,
  nrounds = 254  # Best iterations numbers
)

# Predict probabilities on training set
prob_train <- predict(model_lgb, as.matrix(train_data))

# Predict probabilities on testing set
prob_test <- predict(model_lgb, as.matrix(test_data))

# Compute AUC for training and testing sets
auc_train <- auc(train_label, prob_train)
auc_test <- auc(test_label, prob_test)
print(paste("Train AUC:", round(auc_train, 4)))
print(paste("Test AUC:", round(auc_test, 4)))
roc_train <- roc(train_label, prob_train)

# Determine optimal cutoff based on Youden's index
plot(roc_train,
     legacy.axes = TRUE,
     main="ROC Curve with Optimal Threshold",
     print.thres="best",
     print.auc = TRUE) 

best_cutoff <- 0.511

# Compute training set performance metrics
pred_train <- ifelse(prob_train >= best_cutoff, "2", "1") 
pred_train <- factor(pred_train, levels = levels(data_train$Group))

cm_train <- confusionMatrix(pred_train, data_train$Group, positive = "2")
print(cm_train) 

tp <- cm_train$table["2", "2"]
tn <- cm_train$table["1", "1"]
fp <- cm_train$table["2", "1"]
fn <- cm_train$table["1", "2"]

accuracy <- (tp + tn) / sum(cm_train$table)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
f1 <- 2 * precision * recall / (precision + recall)
npv <- tn / (tn + fn)

# Output performance metrics
cat("Accuracy:", round(accuracy, 6), "\n")
cat("Recall (Sensitivity):", round(recall, 6), "\n")
cat("Precision (PPV):", round(precision, 6), "\n")
cat("NPV:", round(npv, 6), "\n")
cat("F1 Score:", round(f1, 6), "\n")

# Hosmer-Lemeshow test and Brier score for training set
hoslem_train <- hoslem.test(train_label, prob_train, g = 10)
print(hoslem_train)

brier_train <- mean((train_label - prob_train)^2)
cat("Brier Score (Train):", round(brier_train, 6), "\n")

# Compute testing set performance metrics
pred_test <- ifelse(prob_test >= best_cutoff, "2", "1") 
pred_test <- factor(pred_test, levels = levels(data_test$Group))
cm <- confusionMatrix(pred_test, data_test$Group,positive = "2") #Confusion matrix for testing set
cm
roc_test <- roc(response = data_test$Group, predictor = prob_test)
auc(roc_test)  

Tp <- cm$table["2", "2"]
Tn <- cm$table["1", "1"]
Fp <- cm$table["2", "1"]
Fn <- cm$table["1", "2"]

Accuracy <- (Tp + Tn) / sum(cm$table)
Recall <- Tp / (Tp + Fn)
Precision <- Tp / (Tp + Fp)
F1 <- 2 * Precision * Recall / (Precision + Recall)
Npv <- Tn / (Tn + Fn)

# Output testing set performance metrics
cat("Accuracy:", round(Accuracy, 6), "\n")
cat("Recall (Sensitivity):", round(Recall, 6), "\n")
cat("Precision (PPV):", round(Precision, 6), "\n")
cat("NPV:", round(Npv, 6), "\n")
cat("F1 Score:", round(F1, 6), "\n")

# Hosmer-Lemeshow test p-value for testing set
Hl_test <- hoslem.test(test_label, prob_test)
Hl_test$p.value

# Brier Score for testing set
Brier_score <- mean((test_label - prob_test)^2)
cat("Brier Score:", round(Brier_score, 6), "\n")

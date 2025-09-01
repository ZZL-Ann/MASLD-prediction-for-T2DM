# Load required packages
library(xgboost)            # Main package for Extreme Gradient Boosting implementation
library(Matrix)             # Sparse matrix operations
library(caret)              # Cross-validation and model tuning
library(data.table)         # Data handling
library(ResourceSelection)  # Hosmer-Lemeshow goodness-of-fit test


# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Read the dataset from the specified file path
data_test <- read.csv("C:/Users/Desktop/R data/data_test.csv") 

# Selected feature variables
labelVars <- c("Wt","BMI" ,"Lymph","ALB","ALT", "HDL","TG","HbA1c","FCP","Gender","Met","MedStatus","HLP") 
catVars <- c("Gender", "Met", "MedStatus", "HLP")

# Generate contrasts for categorical variables
contrasts_list <- lapply(data_train[, catVars, drop=FALSE], function(x) contrasts(x, contrasts=TRUE))
names(contrasts_list) <- catVars

# Generate model matrices (including intercept, reference-level coding by default)
train_matrix <- model.matrix(~ ., data = data_train[, labelVars], contrasts.arg = contrasts_list)
test_matrix  <- model.matrix(~ ., data = data_test[, labelVars],  contrasts.arg = contrasts_list)

# Target variables encoding (0/1)
train_label <- as.numeric(data_train$Group) - 1
test_label  <- as.numeric(data_test$Group)  - 1

# Construct DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix,  label = test_label)

# Check encoding of categorical variables (example: Gender)
print(colnames(train_matrix)[grepl("Gender", colnames(train_matrix))])


# Define parameter grid for search
param_grid <- expand.grid(
  eta = seq(0.01,0.4,by=0.01),                              # Learning rate
  max_depth = seq(2,10,by = 1),                             # Maximum tree depth
  min_child_weight = c(1,2,3,4,5,6),                        # Minimum sum of instance weight in a child
  gamma = c(seq(0, 1, by = 0.05),3,5),                      # Minimum loss reduction for a split
  subsample = seq(0.05, 1, by = 0.05),                      # Row sampling ratio
  colsample_bytree = c(0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),  # Column sampling ratio per tree
  reg_lambda = seq(0,10,by=0.5)                             # L2 regularization weight
)

# Prepare data frame to store results
results <- data.frame()

# Loop through parameter grid
set.seed(123)
for(i in 1:nrow(param_grid)){
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = param_grid$eta[i],
    max_depth = param_grid$max_depth[i],
    min_child_weight = param_grid$min_child_weight[i],
    gamma = param_grid$gamma[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    reg_lambda = param_grid$reg_lambda[i]
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 1000,                
    nfold = 10,                    # 10-fold cross-validation
    stratified = TRUE,
    early_stopping_rounds = 10,    # Early stopping to prevent overfitting
    maximize = TRUE,
    verbose = 0
  )
  
  best_iter <- cv$best_iteration
  best_auc <- cv$evaluation_log$test_auc_mean[best_iter]
  # Store results
  results <- rbind(results,
                   cbind(param_grid[i,], best_iter = best_iter, test_auc = best_auc))
}

# View the best parameter combination
results <- results[order(-results$test_auc), ]
best_params <- results[1,]
print(best_params)

# Final optimal hyperparameter configuration
base_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.19,
  max_depth = 2,
  min_child_weight = 3,
  gamma = 0.5,
  subsample = 0.95,
  colsample_bytree = 0.75,
  reg_lambda = 2
)

# Train the XGB model
set.seed(123)
xgb_model <- xgb.train(
  params = base_params,
  data = dtrain,
  nrounds = 211)

# Predict probabilities on training set
xgb_prob_train <- predict(xgb_model, newdata = dtrain)
roc_train <- roc(response = train_label, predictor = xgb_prob_train)
cat("AUC:", auc(roc_train), "\n")

# Determine optimal cut-off using Youden's index
plot(roc_train,
     legacy.axes = TRUE,
     main="ROC Curve with Optimal Threshold",
     print.thres="best",
     print.auc = TRUE) 

best_cutoff <- 0.503

# Generate predicted classes based on cut-off
xgb_pred_train <- ifelse(xgb_prob_train >= best_cutoff, "1", "0")

# Confusion matrix and performance metrics
cm_train <- confusionMatrix(factor(xgb_pred_train), factor(train_label), positive = "1")
print(cm_train)
tp <- cm_train$table["1", "1"]
tn <- cm_train$table["0", "0"]
fp <- cm_train$table["1", "0"]
fn <- cm_train$table["0", "1"]

accuracy <- (tp + tn) / sum(cm_train$table)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
f1 <- 2 * precision * recall / (precision + recall)
npv <- tn / (tn + fn)

cat("Accuracy:", round(accuracy, 6), "\n")
cat("Recall (Sensitivity):", round(recall, 6), "\n")
cat("Precision (PPV):", round(precision, 6), "\n")
cat("NPV:", round(npv, 6), "\n")
cat("F1 Score:", round(f1, 6), "\n")

# Hosmer-Lemeshow test for goodness-of-fit
hoslem_train <- hoslem.test(train_label, xgb_prob_train, g = 10)
print(hoslem_train)

# Brier score for training set
brier_train <- mean((train_label - xgb_prob_train)^2)
cat("Brier Score (Train):", round(brier_train, 6), "\n")

# Testing set performance
xgb_prob <- predict(xgb_model, newdata = dtest)  # Predicted probabilities
xgb_pred <- ifelse(xgb_prob >= best_cutoff, "1", "0") # Predicted classes
roc_obj <- roc(response = test_label, predictor = xgb_prob)
auc(roc_obj)  # Output AUC value
cm <- confusionMatrix(factor(xgb_pred), factor(test_label), positive = "1")

tp <- cm$table["1", "1"]
tn <- cm$table["0", "0"]
fp <- cm$table["1", "0"]
fn <- cm$table["0", "1"]

accuracy <- (tp + tn) / sum(cm$table)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
f1 <- 2 * precision * recall / (precision + recall)
npv <- tn / (tn + fn)

cat("Accuracy:", round(accuracy, 6), "\n")
cat("Recall (Sensitivity):", round(recall, 6), "\n")
cat("Precision (PPV):", round(precision, 6), "\n")
cat("NPV:", round(npv, 6), "\n")
cat("F1 Score:", round(f1, 6), "\n")

# Hosmer-Lemeshow test for testing set
hl_test <- hoslem.test(test_label, xgb_prob)
hl_p_value <- hl_test$p.value

# Brier score for testing set
brier_score <- mean((test_label - xgb_prob)^2)
cat("Brier Score:", round(brier_score, 6), "\n")

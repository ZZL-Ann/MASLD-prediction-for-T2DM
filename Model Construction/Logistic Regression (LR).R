# Load required packages
library(glmnet)             # Main package for regularized logistic regression modeling
library(caret)              # Cross-validation and model tuning
library(pROC)               # ROC curve plotting and AUC calculation
library(MLmetrics)          # various ML metrics
library(ResourceSelection)  # Hosmer-Lemeshow goodness-of-fit test
library(DescTools)          # Additional statistical functions

# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Read the training dataset from the specified file path
data_test <- read.csv("C:/Users/Desktop/R data/data_test.csv") # Read the testing dataset

# Selected feature variables
labelVars <- c("Wt","BMI" ,"Lymph","ALB","ALT", "HDL","TG","HbA1c","FCP","Gender","Met","MedStatus","HLP") 

# Create formula for Logistic Regression Model
formula_lr <- as.formula(paste("Group ~", paste(labelVars, collapse = " + ")))

# Keep only relevant variables in training and testing datasets
trainVars <- c("Group", labelVars)
data_train_clean <- data_train[, trainVars]
data_test_clean  <- data_test[, trainVars]

# Set cross-validation control parameters
ctrl <- trainControl(
  method = "cv",                   # 10-fold cross-validation
  number = 10,
  classProbs = TRUE,               # Enable predicted probabilities (for AUC calculation)
  summaryFunction = twoClassSummary,  # Use AUC as evaluation metric
  verboseIter = TRUE)

# Define hyperparameter grid for tuning
grid <- expand.grid(
  alpha = seq(0, 1, 0.05),                # Mixing parameter between Lasso (1) and Ridge (0)
  lambda = 10^seq(log10(0.5), log10(0.001), length.out = 50) )  # Regularization penalty

# Encode outcome variable as factor with specified levels
data_train_clean$Group <- factor(data_train_clean$Group, levels = c("2", "1"), labels = c("with_comorbidity", "no_comorbidity"))

# Train logistic regression model
set.seed(1234)
model <- train(
  formula_lr,
  data = data_train_clean,
  method = "glmnet",           # Use glmnet for regularized logistic regression
  trControl = ctrl,
  tuneGrid = grid,
  metric = "ROC",              # Evaluate using AUC
  family = "binomial",         # logistic regression
  preProcess= c("center", "scale"))         

# Print best combination of hyperparameters
print(model$bestTune)

# Train final logistic regression model using best parameters
best_grid <- expand.grid(alpha = model$bestTune$alpha,lambda = model$bestTune$lambda)
lr_model <- train(
  formula_lr,
  data=data_train_clean,
  method = "glmnet",
  family = "binomial",  
  tuneGrid = best_grid,
  preProcess= c("center", "scale"),
  trControl = trainControl(method = "none"))

# Predict on training set
lr_pred_train <- predict(lr_model, newdata = data_train_clean)
lr_prob_train <- predict(lr_model, newdata = data_train_clean, type = "prob")[, 2]
observed_train <- ifelse(data_train_clean$Group == "2", 1, 0)
observed_test <- ifelse(data_test_clean$Group == "2", 1, 0)

# ROC curve for training set
roc_train <- roc(observed_train, lr_prob_train)

# Determine optimal cut-off threshold using Youden index
plot(roc_train,
     legacy.axes = TRUE,
     main="ROC Curve with Optimal Cut-off",
     print.thres="best", 
     print.auc = TRUE)

best_cutoff <- 0.559

# Compute training set metrics
lr_pred_train_best <- ifelse(lr_prob_train >= best_cutoff, "2", "1") 
lr_pred_train_best <- factor(lr_pred_train_best, levels = levels(data_train_clean$Group))
cm_train <- confusionMatrix(lr_pred_train_best, data_train_clean$Group, positive = "2")
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

# Print training set metrics
cat("Accuracy:", round(accuracy, 6), "\n")
cat("Recall (Sensitivity):", round(recall, 6), "\n")
cat("Precision (PPV):", round(precision, 6), "\n")
cat("NPV:", round(npv, 6), "\n")
cat("F1 Score:", round(f1, 6), "\n")

# Hosmer-Lemeshow test for training set
hoslem_train <- hoslem.test(observed_train, lr_prob_train, g = 10)
print(hoslem_train)

# Brier score for training set
brier_train <- mean((observed_train - lr_prob_train)^2)
cat("Brier Score (Train):", round(brier_train, 6), "\n")

# Compute metrics for testing set
lr_prob <- predict(lr_model, newdata = data_test_clean, type = "prob")[, 2]  
lr_pred <- ifelse(lr_prob >= best_cutoff, "2", "1") 
lr_pred <- factor(lr_pred, levels = levels(data_test_clean$Group))

cm <- confusionMatrix(lr_pred, data_test_clean$Group,positive = "2") # Confusion matrix for testing set
roc_test <- roc(response = data_test_clean$Group, predictor = lr_prob)
auc(roc_test)

tp <- cm$table["2", "2"]
tn <- cm$table["1", "1"]
fp <- cm$table["2", "1"]
fn <- cm$table["1", "2"]

accuracy <- (tp + tn) / sum(cm$table)
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

#Hosmer-Lemeshow Test p-value
observed <- ifelse(data_test_clean$Group == "2", 1, 0)
library(ResourceSelection)
hl_test <- hoslem.test(observed, lr_prob)
hl_p_value <- hl_test$p.value

# Brier score for testing set
brier_score <- mean((observed - lr_prob)^2)
cat("Brier Score:", round(brier_score, 6), "\n")

# Load required packages
library(caret)                 # Cross-validation and model tuning
library(pROC)                  # ROC curve plotting and AUC calculation
library(ResourceSelection)     # Hosmer-Lemeshow goodness-of-fit test
library(MLmetrics)             # Additional ML metrics statistical tools
library(DescTools)             # Extra statistical tools

# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Read the training dataset from the specified file path
data_test <- read.csv("C:/Users/Desktop/R data/data_test.csv") # Read the testing dataset

# Selected feature variables
labelVars <- c("Wt","BMI" ,"Lymph","ALB","ALT", "HDL","TG","HbA1c","FCP","Gender","Met","MedStatus","HLP") 

# Create formula for SVM model
formula_svm <- as.formula(paste("Group ~", paste(labelVars, collapse = " + ")))

# Keep only relevant variables in training and testing datasets
trainVars <- c("Group", labelVars)
data_train_clean <- data_train[, trainVars]
data_test_clean  <- data_test[, trainVars]

# Grid Search Hyperparameter Tuning
# Set training control parameters
ctrl <- trainControl(
  method = "cv",         
  number = 10,            # 10-fold cross-validation
  classProbs = TRUE,     
  summaryFunction = twoClassSummary, 
  savePredictions = "final")

# Create hyperparameter tuning grid (Radial Basis Function kernel)
svm_grid <- expand.grid(
  C = 2^seq(log2(0.1), log2(40), by = 0.5),  # Penalty parameter 
  sigma = 2^seq(-10, -5, by = 0.5))   # Kernel width

# Encode outcome variable as factor with specified levels
data_train_clean$Group <- factor(data_train_clean$Group, levels = c("2", "1"), labels = c("with_comorbidity", "no_comorbidity"))

# Train SVM model
set.seed(123)

model <- train(
  formula_svm,
  data = data_train_clean,
  method = "svmRadial",
  trControl = ctrl,
  preProcess = c("center", "scale"),  
  tuneGrid = svm_grid,
  metric = "ROC")  # Use ROC AUC as evaluation metric

# View the best hyperparameters
model$bestTune
plot(model)

data_train_clean$Group <- data_train$Group 

# Construct best hyperparameter combination
best_grid <- expand.grid(sigma = model$bestTune$sigma,
                         C = model$bestTune$C)

# Train final SVM model with optimal parameters
svm_model <- train(
  formula_svm,
  data = data_train_clean,
  method = "svmRadial",
  tuneGrid = best_grid,
  preProcess = c("center", "scale"),
  trControl = trainControl(method = "none",classProbs = TRUE)  
)  

# Predict probabilities for training set
svm_prob_train <- predict(svm_model, newdata = data_train_clean, type = "prob")[, "with_comorbidity"]

# Restore factor levels for outcome
data_train_clean$Group <- factor(data_train_clean$Group, levels = c("2", "1"))
data_test_clean$Group  <- factor(data_test_clean$Group,  levels = c("2", "1"))

# Create observed labels (1 = positive class "2")
observed_train <- ifelse(data_train_clean$Group == "2", 1, 0)
observed_test  <- ifelse(data_test_clean$Group  == "2", 1, 0)

# ROC curve for training set
roc_train <- roc(observed_train, svm_prob_train)
auc(roc_train, digits = 4) 

#Determine optimal cut-off threshold using Youden's index
plot(roc_train,
     legacy.axes = TRUE,
     main="ROC Curve with Optimal Cut-off",
     print.thres="best", 
     print.auc = TRUE) 

best_cutoff <- 0.578

# Compute training set metrics
svm_pred_train_best <- ifelse(svm_prob_train >= best_cutoff, "2", "1")
svm_pred_train_best <- factor(svm_pred_train_best, levels = c("2", "1"))

cm_train <- confusionMatrix(svm_pred_train_best,
                            data_train_clean$Group,
                            positive = "2")
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
hoslem_train <- hoslem.test(observed_train, svm_prob_train, g = 10)
print(hoslem_train)

# Brier score for training set
brier_train <- mean((observed_train - svm_prob_train)^2)
cat("Brier Score (Train):", round(brier_train, 6), "\n")

# Testing set performance metrics
svm_prob <- predict(svm_model, newdata = data_test_clean,type="prob")[,"with_comorbidity"]
svm_pred <- ifelse(svm_prob >= best_cutoff, "2", "1")
svm_pred <- factor(svm_pred, levels = levels(data_test_clean$Group))
cm <- confusionMatrix(data = svm_pred,                    # Predicted labels
                      reference = data_test_clean$Group)  # True labels
cm
roc_test <- roc(response = data_test_clean$Group , predictor = svm_prob,levels = c("1", "2"))

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

# Hosmer-Lemeshow Test p-value
hl_test <- hoslem.test(observed_test, svm_prob)
hl_p_value <- hl_test$p.value

# Brier score for testing set
brier_score <- mean((observed_test - svm_prob)^2)
cat("Brier Score:", round(brier_score, 6), "\n")

# Load required packages
library(randomForest)      # Main package for Random Forest modeling
library(caret)             # Cross-validation and model tuning
library(pROC)              # ROC curve plotting and AUC calculation
library(ResourceSelection) # Hosmer-Lemeshow goodness-of-fit test

# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Read the dataset from the specified file path
data_test <- read.csv("C:/Users/Desktop/R data/data_test.csv") 

# Selected feature variables
labelVars <- c("Wt","BMI" ,"Lymph","ALB","ALT", "HDL","TG","HbA1c","FCP","Gender","Met","MedStatus","HLP") 

# Create formula for Random Forest model
formula_rf <- as.formula(paste("Group ~", paste(labelVars, collapse = " + ")))

# Keep only relevant variables in training and testing datasets
trainVars <- c("Group", labelVars)
data_train_clean <- data_train[, trainVars]
data_test_clean  <- data_test[, trainVars]

# Grid Search Hyperparameter Tuning
# Custom RF model to allow additional parameters
customRF <- list(
  type = "Classification",
  library = "randomForest",
  loop = NULL,
  parameters = data.frame(parameter = c("mtry", "ntree", "nodesize"),
                          class = rep("numeric", 3),
                          label = c("mtry", "ntree", "nodesize")),
  grid = function(x, y, len = NULL, search = "grid") {},
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    randomForest(x, y,
                 mtry = param$mtry,
                 ntree = param$ntree,
                 nodesize = param$nodesize, ...)
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    predict(modelFit, newdata)
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    predict(modelFit, newdata, type = "prob")
  },
  sort = function(x) x,
  levels = function(x) x$classes
)

# Set cross-validation method and evaluation metric (AUC)
ctrl <- trainControl(
  method = "cv",               
  number = 10,                 # 10-folds Cross-validation
  classProbs = TRUE,           # Enable predicted probabilities (for AUC calculation)
  summaryFunction = twoClassSummary,  
  verboseIter = TRUE           # Print progress for each fold
)

# Define parameter grid using seq()
grid <- expand.grid(
  mtry = seq(1, 13, 1),        # features considered at each split
  ntree = seq(100, 1500, by = 100), # Number of trees
  nodesize = seq(2, 10, 1)     # Minimum node size for each tree
)
set.seed(1234)

# Encode outcome variable as factor with specified levels
data_train_clean$Group <- factor(data_train_clean$Group, levels = c("2", "1"), labels = c("with_comorbidity", "no_comorbidity"))

# Train RF model with grid search and cross-validation
model <- train(
  formula_rf,                   # Formula specifying outcome and predictors
  data = data_train_clean,      
  method = customRF,            # Custom RF model
  tuneGrid = grid,              # Defined hyperparameter grid
  trControl = ctrl,             # Defined cross-validation control
  metric = "ROC"                # AUROC as evaluation metric
)
print(model$bestTune) # Print the best hyperparameters
 
# Train final RF model using selected best hyperparameters
set.seed(1234)
rf_model <- randomForest(formula_rf,
                         data = data_train_clean,
                         ntree = 200,       
                         mtry = floor(sqrt(2)),   
                         importance = TRUE,
                         nodesize =8) 
# Calculate predicted probabilities on training set
rf_prob_train <- predict(rf_model, newdata = data_train_clean, type = "prob")[,2]
roc_train <- roc(data_train_clean$Group, rf_prob_train)

# ROC curve and determine optimal cut-off using Youden index
plot(roc_train,
     legacy.axes = TRUE,
     main="ROC Curve with Optimal Cut-off",
     print.thres="best", 
     print.auc = TRUE) 

best_cutoff <- 0.577

# Compute confusion matrix and classification metrics for training set
rf_pred_train_best <- ifelse(rf_prob_train >= best_cutoff, "2", "1") 
rf_pred_train_best <- factor(rf_pred_train_best, levels = levels(data_train_clean$Group))
cm_train <- confusionMatrix(rf_pred_train_best, data_train_clean$Group, positive = "2")
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
observed_train <- ifelse(data_train_clean$Group == "2", 1, 0)
hoslem_train <- hoslem.test(observed_train, rf_prob_train, g = 10)
print(hoslem_train)

# Brier score for training set
brier_train <- mean((observed_train - rf_prob_train)^2)
cat("Brier Score (Train):", round(brier_train, 6), "\n")

# Compute metrics for testing set
rf_prob <- predict(rf_model, newdata = data_test_clean, type = "prob")[, 2]
rf_pred <- ifelse(rf_prob >= best_cutoff, "2", "1") 
rf_pred <- factor(rf_pred, levels = levels(data_test_clean$Group))
cm_test <- confusionMatrix(rf_pred, data_test_clean$Group,positive = "2") # Confusion matrix for testing set
roc_test <- roc(data_test_clean$Group, rf_prob)
auc(roc_test)

accuracy <- mean(rf_pred == data_test_clean$Group)
precision <- posPredValue(rf_pred, data_test_clean$Group)
recall <- sensitivity(rf_pred, data_test_clean$Group)
npv <- negPredValue(rf_pred, data_test_clean$Group)
ppv <- posPredValue(rf_pred, data_test_clean$Group)
f1_score <- 2 * (precision * recall) / (precision + recall)

observed_test <- ifelse(data_test_clean$Group == "2", 1, 0)
hoslem_test <- hoslem.test(observed_test, rf_prob)
print(hoslem_test)

brier_score <- mean((observed_test - rf_prob)^2)

# Print testing set metrics
cat("Accuracy:", accuracy, "\n")
cat("Recall:", recall, "\n")
cat("NPV:", npv, "\n")
cat("PPV:", ppv, "\n")
cat("F1 Score:", f1_score, "\n")
cat("Hosmer-Lemeshow Test p-value:", hl_p_value, "\n")
cat("Brier Score:", brier_score, "\n")
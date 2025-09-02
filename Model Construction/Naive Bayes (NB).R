# Load required packages
library(klaR)              # Main package for Naive Bayes modeling
library(dplyr)             # Data manipulation
library(pROC)              # ROC curve and AUC calculation
library(caret)             # Model training, tuning, and evaluation
library(ResourceSelection) # Hosmer-Lemeshow goodness-of-fit test

# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Load training dataset from the specified file path
data_test <- read.csv("C:/Users/Desktop/R data/data_test.csv") # Load testing dataset

# Selected feature variables
labelVars <- c("Wt","BMI" ,"Lymph","ALB","ALT", "HDL","TG","HbA1c","FCP","Gender","Met","MedStatus","HLP") 

# Create formula for Naive Bayes model
formula_nb <- as.formula(paste("Group ~", paste(labelVars, collapse = " + ")))

# Define the positive class
data_train$Group <- factor(ifelse(data_train$Group == "2", "with", "without"))
data_test$Group  <- factor(ifelse(data_test$Group == "2", "with", "without"))
data_train$Group <- relevel(data_train$Group, ref = "with")
data_test$Group  <- relevel(data_test$Group, ref = "with")

# Harmonize factor levels across training and testing datasets
factorVars <- c("Gender", "Met", "MedStatus", "HLP", "Group")
for (v in factorVars) {
  all_levels <- levels(factor(c(data_train[[v]], data_test[[v]])))
  data_train[[v]] <- factor(data_train[[v]], levels = all_levels)
  data_test[[v]]  <- factor(data_test[[v]],  levels = all_levels)
}

# Define actual observed labels
observed_train <- ifelse(data_train$Group == "with", 1, 0) 
observed_test <- ifelse(data_test$Group == "with", 1, 0) 

# Cross-validation settings for hyperparameter tuning
ctrl <- trainControl(
  method = "cv",        
  number = 10,          # 10-fold cross-validation
  classProbs = TRUE,    # Enable class probabilities
  summaryFunction = twoClassSummary # Use metrics for binary classification 
)
# Construct hyperparameter search space，tune separately when usekernel = TRUE or FALSE
grid_kernel <- expand.grid(
  usekernel = TRUE, # Use kernel density estimation
  fL = 1, # Laplace smoothing
  adjust = seq(1,4, by=0.1) # Bandwidth adjustment
)

grid_nokernel <- expand.grid(
  usekernel = FALSE, # Not use kernel density estimation
  fL = 1,
  adjust = 1  # Irrelevant when usekernel = FALSE
)

nb_grid <- rbind(grid_kernel, grid_nokernel)

# Model training with hyperparameter tuning
set.seed(123)
model_nb <- train(
  formula_nb,
  data = data_train,
  method = "nb",    # Naive Bayes (from klaR package)
  trControl = ctrl,
  tuneGrid = nb_grid,
  metric = "ROC",   # Use AUC as the optimization metric
  preProcess = NULL            
)

# Inspect model results
print(model_nb)
plot(model_nb)

# Evaluate performance on training set
prob_train <- predict(model_nb, newdata = data_train, type = "prob")[, "with"]
roc_train <- roc(data_train$Group, prob_train, levels = c("with", "without"), direction = ">")
auc(roc_train)

# Determine best cutoff using Youden’s index
best_threshold_youden <- as.numeric(coords(roc_train, "best", best.method = "youden", ret = "threshold"))
print(best_threshold_youden)

# Confusion matrix for training set
pred_train <- ifelse(prob_train >= best_threshold_youden, "with", "without") 
cm_train <- confusionMatrix(factor(pred_train, levels = c("with", "without")),
                            factor(data_train$Group, levels = c("with", "without")),
                            positive = "with")
cm_train

# Extract confusion matrix values
tp <- cm_train$table["with", "with"]
fp <- cm_train$table["with", "without"]   
tn <- cm_train$table["without", "without"]
fn <- cm_train$table["without", "with"]

recall <- tp / (tp + fn)
npv <- tn / (tn + fn)
precision <- tp / (tp + fp)
f1 <- 2 * precision * recall / (precision + recall)

# Perform Hosmer–Lemeshow test and calculate the Brier score
hoslem_test <- hoslem.test(observed_train, prob_train, g = 10)
brier_score <- mean((prob_train - observed_train)^2)

# Output training set performance results
cat("Recall:", recall, "\n")
cat("NPV:", npv, "\n")
cat("Precision:", precision, "\n")
cat("F1-score:", f1, "\n")
print(hoslem_test)
cat("Brier score:", brier_score, "\n")

# Evaluate performance on testing set
prob_test <- predict(model_nb, newdata = data_test, type = "prob")[, "with"] # Predicted probabilities

pred_test <- ifelse(prob_test >= best_threshold_youden, "with", "without") # Predicted classes

roc_test <- roc(response = data_test_scaled$Group, predictor = pred_prob, levels = c("without", "with"))
auc(roc_test)

# Confusion matrix
cm_test <- confusionMatrix(factor(pred_test, levels = c("with", "without")),
                           factor(data_test$Group, levels = c("with", "without")),
                           positive = "with")
cm_test 

# Extract confusion matrix values for testing set
TP <- cm_test$table["with", "with"]
FP <- cm_test$table["with", "without"]  
TN <- cm_test$table["without", "without"]
FN <- cm_test$table["without", "with"]
Recall <- TP / (TP + FN)
Npv <- TN / (TN + FN)
Precision <- TP / (TP + FP)
F1 <- 2 * Precision * Recall / (Precision + Recall)

# Hosmer–Lemeshow test and Brier score for testing set
Hoslem_test <- hoslem.test(observed_test, prob_test, g = 10)
Brier_score <- mean((prob_test - observed_test)^2)

# Output testing set performance results
cat("Recall:", Recall, "\n")
cat("NPV:", Npv, "\n")
cat("Precision:", Precision, "\n")
cat("F1-score:", F1, "\n")
print(Hoslem_test)
cat("Brier score:", Brier_score, "\n")



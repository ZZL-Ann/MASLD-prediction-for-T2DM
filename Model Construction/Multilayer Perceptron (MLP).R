# Load required packages
library(h2o)                 # H2O framework for machine learning
library(ResourceSelection)   # Hosmer-Lemeshow goodness-of-fit test
library(caret)               # ML workflow and evaluation

# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Load training dataset from the specified file path
data_test <- read.csv("C:/Users/Desktop/R data/data_test.csv") # Load testing dataset

# Selected feature variables
labelVars <- c("Wt","BMI" ,"Lymph","ALB","ALT", "HDL","TG","HbA1c","FCP","Gender","Met","MedStatus","HLP") 

# Keep only relevant variables in training and testing datasets
trainVars <- c("Group", labelVars)
data_train <- data_train[, trainVars]
data_test <- data_test[, trainVars]

# Initialize H2O environment
h2o.init()

# Convert datasets to H2O format
train_h2o <- as.h2o(data_train)
test_h2o <- as.h2o(data_test)

# Define response and predictor variables
y <- "Group"  
x <- setdiff(names(data_train), y)  

# Hyperparameter settings
hyper_params <- list(
  hidden = list(c(10),c(20),c(30),c(60),c(16,16),c(24,12),c(32,16),c(32,32),c(64,64),c(128,128)),  # Structure of neurons in hidden layers
  activation = c("Rectifier","Tanh","TanhWithDropout" ,"Maxout"), # Activation functions
  input_dropout_ratio = c(0.1,0.2,0.3,0.4),  # Dropout rate
  epochs = c(5,10,15,20,30,40,50),  # Training epochs
  rate = c(0.001,0.005, 0.01, 0.05,0.1)  # Initial learning rate
)

# Perform GridSearch for hyperparameter tuning with cross-validation
grid <- h2o.grid(
  algorithm = "deeplearning", 
  grid_id = "dl_grid_1", 
  x = x, 
  y = y, 
  training_frame = train_h2o, 
  hyper_params = hyper_params,
  search_criteria = list(strategy = "Cartesian"),  # Use Cartesian grid search
  nfolds = 10,  # Number of CV folds
  seed = 123,  # Random seed for reproducibility
  stopping_metric = "AUC",  # Use AUROC as stopping standard reference metric
  stopping_tolerance = 0.0001,  # Stop if improvement < 0.0001
  stopping_rounds = 10,
  distribution = "bernoulli",
  adaptive_rate = TRUE
)

# Retrieve and sort grid search results by AUROC
grid_perf <- h2o.getGrid(grid_id = "dl_grid_1", sort_by = "auc", decreasing = TRUE)
grid_table <- as.data.frame(grid_perf@summary_table)
head(grid_table,20)  # View the top 20 optimal hyperparameter combinations by AUROC

# Train MLP model with selected hyperparameters
mlp_model <- h2o.deeplearning(
  x = x,             
  y = y,              
  training_frame = train_h2o, 
  activation = "Tanh",   
  hidden = c(30),
  epochs = 20,
  input_dropout_ratio = 0.2,
  adaptive_rate = TRUE,
  seed = 123,
  reproducible=TRUE,
  stopping_tolerance = 0.0001, 
  stopping_rounds = 10)

# Training set performance
pred_train <- as.data.frame(h2o.predict(mlp_model, train_h2o))
perf_train <- h2o.performance(mlp_model, newdata = train_h2o)
print(perf_train)
h2o.auc(perf_train) 

# Obtain full threshold-performance table
metrics_df <- as.data.frame(h2o.metric(perf_train))
# Add Youden index column
metrics_df$youden <- metrics_df$tpr + (1 - metrics_df$fpr) - 1
# Find threshold corresponding to maximum Youden index
best_row <- metrics_df[which.max(metrics_df$youden), ]
best_threshold_youden <- best_row$threshold
print(best_row)

# Generate confusion matrix based on optimal threshold
cm_train <- h2o.confusionMatrix(perf_train, thresholds = best_threshold_youden)
print(cm_train)
# Fill in true positives, false positives, etc.
TP <- 1135
FP <- 202
TN <- 984
FN <- 364
N <- TP + FP + TN + FN  # Total samples
accuracy <- (TP + TN) / N  
recall <- TP / (TP + FN)   
precision <- TP / (TP + FP) 
npv <- TN / (TN + FN)      
f1 <- 2 * precision * recall / (precision + recall)  

list(
  Accuracy = round(accuracy, 6),
  Recall = round(recall, 6),
  Precision = round(precision, 6),
  NPV = round(npv, 6),
  F1 = round(f1, 6)
)

# Hosmer-Lemeshow test and Brier score
prob_train <- pred_train$p2 # Predicted probability for positive class
observed_train <-  ifelse(data_train$Group == "2", 1, 0) # Actual labels

hl_test <- hoslem.test(observed_train, prob_train, g = 10)
print(hl_test)
brier_score <- mean((prob_train - observed_train)^2)
cat("Brier Score:", round(brier_score, 6), "\n")

# Testing set performance
pred <- as.data.frame(h2o.predict(mlp_model, test_h2o))  # Predicted probabilities as data frame
perf <- h2o.performance(mlp_model, newdata = test_h2o)   # Obtain testing set performance
print(perf)
h2o.auc(perf) 

# Generate confusion matrix based on training-set optimal threshold
pred_test <- ifelse(pred$p2 >= best_threshold_youden, "2", "1") # Predicted class
true_labels <- factor(data_test$Group, levels = c("1", "2")) # Ensure factor levels match
pred_labels <- factor(pred_test, levels = c("1", "2"))

cm <- confusionMatrix(pred_labels, true_labels, positive = "2")
print(cm)
cm_table <- cm$table
# Extract TP, FP, FN, TN from confusion matrix
tp <- cm_table["2", "2"]  
fp <- cm_table["2", "1"]  
fn <- cm_table["1", "2"]  
tn <- cm_table["1", "1"]  

# Calculate performance metrics
accuracy <- (tp + tn) / (tp + tn + fp + fn)  
precision <- tp / (tp + fp)  
recall <- tp / (tp + fn)  
f1_score <- 2 * (precision * recall) / (precision + recall)  
npv <- tn / (tn + fn)

cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")
cat("NPV:", round(npv, 6), "\n")

# Hosmer-Lemeshow test and Brier score for testing set
prob_test <- pred$p2 
observed_test <-  ifelse(data_test$Group == "2", 1, 0) 
hl_test <- hoslem.test(observed_test, prob_test, g = 10)
print(hl_test)

brier_score <- mean((prob_test - observed_test)^2)
cat("Brier Score:", round(brier_score, 6), "\n")

# End H2O session
h2o.shutdown(prompt = FALSE)
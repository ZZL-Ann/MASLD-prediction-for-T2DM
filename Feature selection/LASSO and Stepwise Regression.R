# Loading required R packages
library(dplyr)       # Data manipulation
library(glmnet)      # LASSO and elastic-net regression
library(Matrix)      # Matrix operations
library(rms)         # Regression modeling strategies
library(foreign)     # Reading data from other formats
library(reader)      # Data reading utility
library(car)         # Variance inflation factor (VIF) calculation
library(MASS)        # Stepwise regression (stepAIC)

# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Read the dataset from the specified file path

# Define categorical and numeric variables
catVars <- c("Gender","AP","Satin","AGI","GLP_1RA","SGLT_2i","DPP_4i","SU","Met","Insulin","Glinides","MedStatus","DKD","DR","DN","DMD","DFU","HTN","HLP","HUA","Group") 
numVars <- c("Age","Wt","Ht","BMI","SBP","DBP","Neut","WBC","Mono","Lymph","Lymph_percent","Eos","Baso","PLT","ALB","ALT","AST","ALP","GGT","TBIL","TC","HDL","LDL","TG","ApoAI","ApoB","HbA1c","FBG","GLU_2h","FINS","FCP","TSH","FT4","FT3","Duration")

data_train[numVars] <- lapply(data_train[numVars], function(x) as.numeric(as.character(x))) # Convert continuous variables to numeric
data_train[catVars] <- lapply(data_train[catVars],factor) # Convert categorical variables to factor
# Log-transform variables with strong right-skew (skewness > 1)
data_train_log <- data_train %>%
  mutate(across(all_of(numVars_noboth), ~ log(pmax(., 1e-6)))) # Replace 0 values (indicating a test value below the lower limit) with a very small number for log-transform

# Perform Z-score standardization
data_Z <- data_train_log
data_Z[, sapply(data_train_log, is.numeric)] <- scale(data_train_log[, sapply(data_train_log, is.numeric)], center = TRUE, scale = TRUE)

# LASSO（Least Absolute Shrinkage and Selection Operate）regression for feature selection
# Prepare independent (x) and dependent (y) matrices
catVars1 <- catVars[1:20] 

# Ensure binary factors have reference level "0" and target level "1"
for (var in catVars1) {
  if (length(levels(data_Z[[var]])) == 2) {
    data_Z[[var]] <- factor(data_Z[[var]], levels = c("0", "1"))}}

# Convert categorical variables to dummy variables
model_catVars <- model.matrix(~ . , data = data_Z[, catVars1])
model_catVars <- model_catVars[, -1] # Remove intercept column

x <- as.matrix(data.frame(data_Z[,numVars],model_catVars)) 
y <- as.numeric(as.matrix(data_Z[, 57])) - 1

# LASSO regression with 10-fold CV
set.seed(0321) # Reproducibility
cv.fit <- cv.glmnet(x, y, alpha = 1, nfolds = 10, family = 'binomial') # LASSO
fit <- glmnet(x, y, alpha = 1, family = 'binomial')

plot(cv.fit)  ## Plot CV results

# Extract coefficients at lambda.1se
cv.fit$lambda.1se  # Select the largest lambda within one standard error of the minimum deviance
Coefficients1se <- coef(fit,s=cv.fit$lambda.1se)
Active.Index1se <- which(Coefficients1se!=0)  
Active.Coefficients1se <- Coefficients1se[Active.Index1se]
Active.Coefficients1se # View the coefficients of selected variables at the optimal lambda
row.names(Coefficients1se)[Active.Index1se] #View the variable names corresponding to non-zero coefficients


# Multicollinearity check using VIF
selected_vars <- c("Wt","BMI" ,"Lymph","Lymph_percent","PLT","ALB","ALT", "GGT",
                   "HDL","TG","HbA1c","GLU_2h","FINS","FCP","Gender","SGLT_2i",
                   "Met","Insulin","MedStatus","DMD","HLP" ) # Variables with non-zero coefficients selected by LASSO regression
vif_model <- glm(data_train$Group ~ ., data = data_selected,family = binomial)  
vif_values <- vif(vif_model)  
print(vif_values) # View the VIF value

# Stepwise logistic regression (BIC)
stepwise_model <- glm(data_train$Group ~ ., data = data_selected,family = binomial) 
reduced_model <- stepAIC(stepwise_model, direction = "both", k = log(nrow(data_train))) # BIC-based selection
summary(reduced_model)
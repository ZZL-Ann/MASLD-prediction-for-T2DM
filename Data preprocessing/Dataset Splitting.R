# Read the dataset from the specified file path
dataAll <- readxl::read_xlsx("C:/Users/Desktop/Total NAFLD Data (Unimpaired).xlsx")

# All variable names
varsAll <- c("Gender", "Age","Wt","Ht","BMI","SBP","DBP","Neut","WBC","Mono","Lymph","Lymph_percent","Eos","Baso","PLT","ALB","ALT","AST","ALP","GGT","TBIL","TC","HDL","LDL","TG","ApoAI","ApoB","HbA1c","FBG","GLU_2h","FINS","FCP","TSH","FT4","FT3","AP","Satin","AGI","GLP_1RA","SGLT_2i","DPP_4i","SU","Met","Insulin","Glinides","Duration","MedStatus","DKD","DR","DN","DMD","DFU","HTN","HLP","HUA","Group")
# Categorical variables
catVars <- c("Gender","AP","Satin","AGI","GLP_1RA","SGLT_2i","DPP_4i","SU","Met","Insulin","Glinides","MedStatus","DKD","DR","DN","DMD","DFU","HTN","HLP","HUA","Group")
# Continuous variables
numVars <- c("Age","Wt","Ht","BMI","SBP","DBP","Neut","WBC","Mono","Lymph","Lymph_percent","Eos","Baso","PLT","ALB","ALT","AST","ALP","GGT","TBIL","TC","HDL","LDL","TG","ApoAI","ApoB","HbA1c","FBG","GLU_2h","FINS","FCP","TSH","FT4","FT3","Duration")
# Convert continuous variables to numeric type
dataAll[numVars] <- lapply(dataAll[numVars], function(x) as.numeric(as.character(x)))
# Convert categorical variables to factors
dataAll[catVars] <- lapply(dataAll[catVars],factor)
# Check data types
str(dataAll) 

# Split the dataset
# Load the necessary packages
library(dplyr)

# Set the random seed number to ensure reproducibility  
set.seed(123)

# Use the sample function to randomly select 70% of the samples from the ID column of the dataAll dataset, and store the sampled results in the train variable.
train <- sample(dataAll$ID,floor(nrow(dataAll)*0.7),replace = F)

# Obtain the sampled train data using the filter function, and split the data into a training cohort (data_train) and a testing cohort (data_train).
data_train <- filter(dataAll,dataAll$ID %in% train)
data_test <- filter(dataAll,!dataAll$ID %in% train)
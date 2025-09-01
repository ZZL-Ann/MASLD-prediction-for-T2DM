# Load the necessary R packages
library(pmsampsize) # Calculate sample size

# Enter estimated incidence 
prevalence <- 0.65

# Compute max(R²_cs) using the formula
lnL_null <- prevalence*log(prevalence) + (1 - prevalence) * log(1 - prevalence) # lnL_null: log-likelihood of an intercept-only model

# Calculate the theoretical maximum value of Cox-Snell R² according to the formula
max_R2_cs <- 1-exp(2 * lnL_null)

# If we conservatively assume the new model will explain 15% of the variance, calculate the expected R² cs value based on this assumption:
rsquared <- 0.15*max_R2_cs 
rsquared

# Use the pmsampsize package to calculate sample size
pmsampsize(
  type="b", #"b" for Binary categorical outcome
  csrsquared = rsquared, # Calculated expected R2CS value
  parameters = 15, # Maximum Estimated Candidate Predictor Variable Count
  prevalence = prevalence, # Incidence of time to outcome in the population
  seed = 12345)

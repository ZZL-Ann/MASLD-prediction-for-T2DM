# Loading required R packages
library(tidyverse)     # Data manipulation and visualization
library(Boruta)        # Boruta feature selection
library(randomForest)  # Random forest
library(ggridges)      # Ridge plot visualization
library(viridis)       # Color palettes
library(scales)        # Transparency control

# Load the dataset
data_train <- read.csv("C:/Users/Desktop/R data/data_train.csv") # Read the dataset from the specified file path

# Define categorical and numeric variables
catVars <- c("Gender","AP","Satin","AGI","GLP_1RA","SGLT_2i","DPP_4i","SU","Met","Insulin","Glinides","MedStatus","DKD","DR","DN","DMD","DFU","HTN","HLP","HUA","Group") 
numVars <- c("Age","Wt","Ht","BMI","SBP","DBP","Neut","WBC","Mono","Lymph","Lymph_percent","Eos","Baso","PLT","ALB","ALT","AST","ALP","GGT","TBIL","TC","HDL","LDL","TG","ApoAI","ApoB","HbA1c","FBG","GLU_2h","FINS","FCP","TSH","FT4","FT3","Duration")

# Prepare independent (x) and dependent (y) matrices
catVars1 <- catVars[1:20]
Borutamodel_catVars <- model.matrix(~ . , data = data_train[, catVars1])
Borutamodel_catVars <- model_catVars[, -1] # Remove intercept column
x_Boruta <- as.matrix(data.frame(data_train[, numVars], Borutamodel_catVars))
y_Boruta <- factor(data_train_log$Group, levels = c("1", "2"))

# Perform Boruta feature selection
boruta_selection <- Boruta(
  x = x_Boruta,
  y = y_Boruta,
  pValue = 0.01,         # Significance level: only variables with p < 0.01 considered important
  mcAdj = TRUE,          # Perform multiple comparison adjustment (Monte Carlo adjustment)
  maxRuns = 100,         # Maximum iterations to prevent infinite loop
  doTrace = 3,           # Print detailed progress
  holdHistory = TRUE,    # Keep importance history for later analysis
  getImp = getImpRfZ     # Variable importance method: Z-score from random forest
)

# Extract feature importance history
imp_history <- as.data.frame(boruta_selection$ImpHistory)

# Identify shadow feature columns
shadow_cols <- grep("shadow", colnames(imp_history), value = TRUE)

# Identify real feature columns (excluding shadow features)
real_cols <- setdiff(colnames(imp_history), shadow_cols)

# Create decision table for real features
final_decision <- tibble(
  feature = real_cols,
  decision = boruta_selection$finalDecision[real_cols]
)

# Create shadow decision table
shadow_decision <- tibble(
  feature = shadow_cols,
  decision = rep("shadow", length(shadow_cols))
)

# Combine real and shadow decisions
decision_all <- bind_rows(final_decision, shadow_decision)

# Convert imp_history to long format and merge with Boruta decisions
imp_history_long <- imp_history %>%
  pivot_longer(
    cols = everything(),
    names_to = "feature",
    values_to = "importance"
  ) %>%
  left_join(decision_all, by = "feature")

# Filter out non-finite values (NA or Inf) to ensure data is suitable for plotting
imp_history_long_clean <- imp_history_long %>% filter(is.finite(importance))

# Determine feature order by median importance for plotting
feature_order <- imp_history_long_clean %>%
  group_by(feature) %>%
  summarise(median_imp = median(importance, na.rm = TRUE)) %>%
  arrange(median_imp) %>%
  pull(feature)

# Plot feature importance ridge plot
ggplot(imp_history_long_clean, aes(
  x = importance,                      # Importance (Z-score)
  y = fct_relevel(feature, feature_order), # Feature order by median importance
  fill = decision                      # Fill by decision category
)) +
  geom_density_ridges_gradient(
    scale = 3,                          # Ridge height scaling
    rel_min_height = 0.01,              # Minimum relative height to avoid tiny peaks
    color = "black"                      # Ridge border color
  ) +
  scale_fill_manual(
    values = c(
      "Confirmed" = alpha("#02BBC1", 0.6), # Confirmed important
      "Tentative" = alpha("#FFC107", 0.6), # Tentative
      "Rejected"  = alpha("#E53953", 0.6), # Rejected
      "shadow"    = alpha("#757575", 0.6)  # Shadow features
    )
  ) +
  labs(
    title = "Feature Importance Ridge Plot Based on Boruta",
    x = "Importance (Z-score)",
    y = "Features"
  ) +
  theme_bw() +
  theme(
    legend.position = "right",
    panel.spacing = unit(0.1, "lines"),
    axis.text.y = element_text(size = 10)
  )
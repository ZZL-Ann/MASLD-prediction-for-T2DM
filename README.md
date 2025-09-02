# MASLD-prediction-for-T2DM
1.Project Overview
   -
&nbsp;This repository contains the core scripts for developing machine learning models to predict Metabolic Associated Steatotic Liver Disease (MASLD) in patients with Type 2 Diabetes Mellitus (T2DM), based on routinely collected clinical and laboratory data. We constructed and compared eight machine learning algorithms. The project includes data preprocessing, feature selection, and predictive modeling.The project structure is as follows:  
├── README.md  
├── Data preprocessing/  
│   ├── Dataset Splitting.R  
│   ├── Multiple Imputation.R  
│   └── Sample size Calculation  
├── Feature selection/  
│   ├── LASSO and Stepwise Regression.R  
│   └── Boruta.R  
└── Model Construction/  
&nbsp;&nbsp;    ├── Random Forest (RF).R  
&nbsp;&nbsp;    ├── Logistic Regression (LR).R  
&nbsp;&nbsp;    ├── K-Nearest Neighbors (KNN).R  
&nbsp;&nbsp;    ├── Extreme Gradient Boosting (XGB).R  
&nbsp;&nbsp;    ├── Multilayer Perceptron (MLP).R  
&nbsp;&nbsp;    ├── Naive Bayes (NB).R  
&nbsp;&nbsp;    ├── Support Vector Machine (SVM).R  
&nbsp;&nbsp;    └── Light Gradient Boosting Machine (LightGBM).R  

2.Methods
   -

&nbsp;2.1 Data preprocessing  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Factor levels are unified across training and testing datasets.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Outcome variable (Group) is binarized:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·T2DM with MASLD (Group = 2)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·T2DM without MASLD (Group = 1)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Continuous and categorical variables are handled appropriately for each algorithm including necessary standardization and normalization.  

&nbsp;2.2 Feature Selection  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We employed LASSO combined with stepwise multivariate logistic regression based on the Bayesian Information Criterion to screen feature variables from 55 candidate variables and additionally validated the feature variables using the independently run Boruta algorithm.  

&nbsp;2.3 Model Construction  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We employed eight widely used machine learning (ML) algorithms:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.1 Random Forest  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model principle: Ensemble of decision trees trained on bootstrap samples with random feature selection; reduces variance and improves generalization.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: ntree, mtry, nodesize.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.2 Logistic Regression  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model principle: Linear model for binary classification using logit link; estimates probability of positive class via linear combination of features.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: alpha, lambda.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.3 K-Nearest Neighbors  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model principle: Non-parametric; predicts class based on majority vote among k nearest neighbors in feature space.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: k, distance metric, kernal.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.4 Extreme Gradient Boosting  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model principle: Gradient boosting framework that sequentially builds trees to minimize a differentiable loss function; incorporates regularization to prevent overfitting.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: max_depth, eta (learning_rate), regularization (lambda, alpha), sampling (subsample, colsample_bytree), nrounds.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.5 Multilayer Perceptron  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model principle: Feedforward neural network with one or more hidden layers; learns non-linear relationships between features and outcome using backpropagation.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: hiden layzer sizes, activation, learning rate, epoch, input dropout ratio.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.6 Naive Bayes  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model principle: Probabilistic classifier based on Bayes’ theorem; assumes conditional independence between features given the class.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: usekernel, fL (Laplace correction), adjust (kernel bandwidth).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.7 Support Vector Machine  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model principle:Finds optimal hyperplane to separate classes; use kernel functions to handle non-linear separations.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: kernel, C (regularization), gamma (RBF kernel parameter).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.8 Light Gradient Boosting Machine  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model principle: Gradient boosting framework based on decision trees; uses histogram-based algorithms and leaf-wise growth for faster training and better accuracy.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: num_leaves, max_depth, min_data_in_leaf, learning_rate, regularization (lambda_l1, lambda_l2), sampling (bagging_fraction, feature_fraction).  
	 
3.Model Evaluation
   -
&nbsp;3.1 ROC curves are plotted to evaluate the model's discrimination capability, and the optimal cutoff is determined using maximum Youden's index.  

&nbsp;3.2 Performance metrics (Accuracy,Recall,NPV,PPV/Precision,F1 score) calculated for both training and testing datasets.  

&nbsp;3.3 Calibration assessed with Hosmer-Lemeshow test and Brier score.  

4.Usage
-
This project is implemented in R. Each step of the workflow has its corresponding script located in the folder with the same name. All required R packages are listed at the beginning of each script and should be installed prior to running the code.

Note: Some models, such as the Multi-Layer Perceptron (MLP), require a running Java environment within R (rJava) to execute properly. Make sure Java is installed and configured in your system before running these scripts.

To run the project:

&nbsp;&nbsp;  1.Open the R script for the specific model or analysis step.

&nbsp;&nbsp;  2.Install and load the required packages if not already installed.

&nbsp;&nbsp;  3.Execute the script sequentially as indicated in the comments.

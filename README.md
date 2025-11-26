# CMPT 310 Project: Housing Affordability Analysis in British Columbia
---
Python Libs:

```
Python        3.11.x
JupyterLab    4.4.9
numpy         2.3.3
pandas        2.3.3
matplotlib    3.10.6
seaborn       0.13.2
scikit-learn  1.7.2
```

Setup (Conda ENV):

```
# Conda env setup
conda create -n housing_pred python=3.11 -y
conda activate housing_pred
conda install jupyterlab=4.4.9 -y
conda install numpy=2.3.3 pandas=2.3.3 matplotlib=3.10.6 seaborn=0.13.2 scikit-learn=1.7.2 -y

# Verify installations
python -m pip list
jupyter lab --version
```

***See requirements.txt for exact env setup***

---

## Project Description

This project focuses on analyzing and predicting housing affordability in British Columbia using a **quarterly time-series dataset**. The dataset contains detailed housing market and economic indicators.

Because the data spans multiple years in quarterly intervals, it captures **long-term trends** and **seasonal patterns**, making it suitable for applying regression models to predict either:

1. **Median housing prices**  
2. **Housing affordability**, defined as the payment-to-income ratio

---

## Project Workflow

### 1. Data Exploration and Feature Analysis

- Conducted correlation studies to identify relationships between features and target variables  
- Generated visualizations such as heatmaps and scatter plots to summarize patterns and feature redundancy  
- Verified expected relationships, e.g., lower prime interest rates correlating with higher housing prices  

### 2. Regression Model Benchmarking

- Established a baseline using **Linear Regression**  
- Evaluated multiple linear models: RidgeCV, LassoCV, HuberRegressor, SGDRegressor, and ElasticNetCV  
- Tested polynomial features to check for non-linear trends; found that the housing price trend is largely linear  

### 3. Temporal Evaluation

- Applied temporal splits (e.g., 2010 and 2018) to simulate real-world prediction scenarios  
- Compared models based on **MSE** and **R²**, visualizing performance across models  
- Found that model performance is highly sensitive to the choice of training and testing periods  

### 4. Gradient Boosting with Residuals

To capture both linear trends and non-linear patterns:

1. **Trend features** (linear relationships) are modeled using a linear regression to predict housing prices.  
2. **Residuals** are calculated as the differences between actual prices and the linear trend predictions.  
3. A **Gradient Boosting Regressor** with decision trees as base learners is trained on **tree features** to predict these residuals.  
4. **Final predictions** are obtained by adding the predicted residuals to the linear trend predictions, combining linear extrapolation with non-linear adjustments.

Key points:

- Trend features: all features except Unemployment Rates  
- Tree features: Unemployment Rates, Prime Interest Rate, Historical Average Payment to Income Percent  
- R² score before boosting (linear trend only): 0.80  
- R² score after boosting (residual adjustment): 0.87  
- Baseline linear regression R²: 0.78  

Challenges included selecting appropriate trend and tree features. Initial attempts based on feature collinearity were insufficient, so a more experimental, manual approach was used. Some features, like Prime Interest Rate and Historical Average Payment to Income Percent, improved accuracy when included in both trend and tree sets.

---

## Key Features

- **Data Visualizations:** Correlation heatmaps, feature-target relationships, model performance charts  
- **Regression Models:** Linear, regularized, and robust regressors; Gradient Boosting for residual modeling  
- **Temporal Splitting:** Evaluates model generalization across different historical periods  
- **Reproducibility:** Python 3.11.x compatible, with scripts for data cleaning, analysis, and model evaluation  

---

## Project Goals

- Understand the relationships between economic, demographic, and housing features  
- Identify which features most strongly influence housing prices and affordability  
- Compare the performance of multiple regression models in predicting future housing trends  
- Develop a workflow for time-series prediction and feature analysis in the housing domain  

---

This repository contains all scripts, datasets, and notebooks necessary to reproduce the analyses, generate visualizations, and evaluate model performance.

# 🏠 House Price Prediction

### End-to-End Machine Learning Regression Project

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Regression-FFCC00)](https://catboost.ai/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas\&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy\&logoColor=white)](https://numpy.org/)

> A machine learning project focused on predicting residential house prices using structured property data.
> The project covers the complete modeling workflow — from data exploration and preprocessing to feature handling, cross-validation, model training and final prediction generation.

---

## 📌 Project Overview

The objective of this project is to build a robust regression model capable of estimating house sale prices from a wide range of property characteristics.

The project follows an end-to-end machine learning workflow:

```text
Raw Data
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Data Cleaning
   │
   ▼
Missing Value Treatment
   │
   ▼
Outlier Analysis
   │
   ▼
Target Transformation
   │
   ▼
Feature Preparation
   │
   ▼
Cross-Validation
   │
   ▼
CatBoost Regression
   │
   ▼
Model Evaluation
   │
   ▼
Final Model
   │
   ▼
House Price Predictions
```

The main model is based on **CatBoostRegressor**, a gradient boosting algorithm particularly well suited for structured/tabular datasets containing both numerical and categorical variables.

---

# 🎯 Objective

The main goal is to minimize prediction error while building a model that generalizes well to unseen housing data.

The project focuses on:

* understanding the structure of the housing dataset,
* identifying missing and problematic observations,
* analyzing feature distributions,
* handling numerical and categorical variables,
* reducing the influence of extreme observations,
* transforming the target variable,
* validating model performance using cross-validation,
* training a final regression model,
* generating predictions for unseen data.

---

# 📊 Dataset

The project uses separate training and test datasets:

```text
train.csv
test.csv
```

The training dataset contains the target variable used for supervised learning, while the test dataset is used to generate final predictions.

The feature space contains both:

* numerical variables,
* categorical variables.

This combination makes CatBoost a suitable choice because it can work directly with categorical features without requiring traditional one-hot encoding for every category.

---

# 🔬 Methodology

## 1. Exploratory Data Analysis

The first stage focuses on understanding the dataset before model training.

The analysis includes:

* feature distributions,
* categorical variables,
* missing values,
* target distribution,
* potential outliers,
* relationships between variables.

Additional diagnostic visualizations are included in the repository, such as distribution analysis and Cook's distance analysis.

---

## 2. Missing Value Treatment

Missing values are handled differently depending on variable type.

### Numerical features

Missing numerical values are replaced using the **median calculated from the training data**.

```text
Missing numerical value
        ↓
Training-set median
        ↓
Imputed value
```

### Categorical features

Missing categorical observations are explicitly represented as:

```text
"Missing"
```

This allows CatBoost to treat missing categorical information as an additional category instead of discarding observations.

---

# 📈 3. Target Transformation

The target variable is transformed using a logarithmic transformation:

```python
y = np.log1p(y)
```

This approach is useful for house-price prediction because property prices commonly exhibit a positively skewed distribution.

The transformation helps:

* reduce skewness,
* reduce the influence of extremely expensive properties,
* stabilize variance,
* make the regression problem easier for the model to learn.

---

# ⚠️ 4. Outlier Analysis

Extreme observations are explicitly investigated rather than blindly removed.

The project identifies observations with unusually large living areas combined with very high target values.

These observations are then excluded from the training set when they meet the defined outlier criteria.

This step is designed to prevent a small number of extreme properties from disproportionately influencing the regression model.

---

# 🧠 5. Model — CatBoost Regressor

The primary model is:

```text
CatBoostRegressor
```

The implemented configuration includes:

| Parameter              | Value |
| ---------------------- | ----: |
| Iterations             |  8000 |
| Learning Rate          |  0.01 |
| Depth                  |     5 |
| Loss Function          |  RMSE |
| L2 Leaf Regularization |     8 |
| Random Strength        |   0.5 |
| Bagging Temperature    |   0.2 |
| Random Seed            |    42 |

CatBoost was selected because it is designed for high-performance gradient boosting on tabular datasets and provides native support for categorical features.

---

# 🔁 6. Cross-Validation

Model performance is evaluated using **10-fold K-Fold Cross-Validation**.

```text
Dataset
   │
   ├── Fold 1 → Train / Validation
   ├── Fold 2 → Train / Validation
   ├── Fold 3 → Train / Validation
   ├── ...
   └── Fold 10 → Train / Validation
```

Configuration:

```python
KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
```

For every fold, the model is trained on the training portion and evaluated on the validation portion.

The individual R² scores are then aggregated to calculate:

* mean R²,
* standard deviation of R².

This provides a more reliable estimate of model stability than relying on a single train/test split.

---

# ⏹️ 7. Early Stopping

During cross-validation, the model uses an evaluation set and early stopping.

The training process can therefore stop when additional boosting iterations no longer improve validation performance.

This helps reduce unnecessary training and limits overfitting.

---

# 🏆 8. Final Model

After cross-validation, a final CatBoost model is trained using the prepared dataset.

The resulting trained model is serialized and stored as:

```text
house_price_model.pkl
```

This file contains the trained machine learning model and can be used later for generating predictions without repeating the entire training process.

---

# 📤 9. Predictions

Predictions are generated for the unseen test dataset.

The repository contains generated prediction files:

```text
submission.csv
submission_catboost.csv
```

These files represent the model's predicted house prices for the test observations.

---

# 📁 Project Structure

```text
house_price_model/
│
├── .idea/
│
├── catboost_info/
│
├── cat.py
│   └── CatBoost model training pipeline
│
├── data.py
│   └── Data exploration and preprocessing
│
├── train.csv
│   └── Training dataset
│
├── test.csv
│   └── Test dataset
│
├── house_price_model.pkl
│   └── Serialized trained model
│
├── submission.csv
│   └── Generated predictions
│
├── submission_catboost.csv
│   └── CatBoost predictions
│
├── rozklady_cech.png
│   └── Feature distribution visualization
│
└── cooks.png
    └── Outlier / influence diagnostics
```

---

# 🛠️ Tech Stack

| Technology                       | Purpose                                         |
| -------------------------------- | ----------------------------------------------- |
| **Python**                       | Core programming language                       |
| **Pandas**                       | Data manipulation and preprocessing             |
| **NumPy**                        | Numerical computations                          |
| **Scikit-learn**                 | Cross-validation and machine learning utilities |
| **CatBoost**                     | Gradient boosting regression                    |
| **Matplotlib**                   | Data visualization                              |
| **Jupyter / Python environment** | Model development and experimentation           |
| **Pickle**                       | Model serialization                             |

---

# 📐 Machine Learning Pipeline

The complete pipeline can be summarized as:

```text
                    ┌─────────────────┐
                    │   Housing Data  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │       EDA       │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Missing Value       │
                  │ Treatment           │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Outlier Analysis    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Target Log Transform│
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ CatBoost Regression │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 10-Fold CV          │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Model Evaluation    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Final Model         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Price Predictions   │
                  └─────────────────────┘
```

---

# 📊 Model Evaluation

The primary validation metric used during the CatBoost cross-validation process is **R² (coefficient of determination)**.

For each validation fold:

```text
Fold 1 → R²
Fold 2 → R²
Fold 3 → R²
...
Fold 10 → R²
```

The project then calculates:

```text
Mean R²
Standard Deviation
```

This makes it possible to evaluate not only predictive performance but also the stability of the model across different validation subsets.

> The exact validation results are generated during model execution and should be interpreted together with the fold-level scores and their variance.

---

# 🔎 Model Diagnostics

The repository also contains additional diagnostic analysis.

### Feature distributions

`rozklady_cech.png`

Used to inspect the distribution of selected variables and identify potentially skewed or unusual features.

### Influence / outlier diagnostics

`cooks.png`

Used to investigate observations that may have an unusually large influence on the fitted regression problem.

These diagnostics complement the numerical evaluation and help identify potential data-quality issues.

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/Drwalpawel20/house_price_model.git
cd house_price_model
```

## 2. Install dependencies

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install pandas numpy scikit-learn catboost matplotlib
```

## 3. Train the model

Run:

```bash
python cat.py
```

The training process performs preprocessing, cross-validation and final model training.

---

# 💾 Saved Model

The trained model is stored as:

```text
house_price_model.pkl
```

This allows the model to be reused without retraining.

---

# 📌 Key Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* Supervised Learning
* Regression
* Gradient Boosting
* CatBoost
* Feature preprocessing
* Missing value imputation
* Categorical feature handling
* Logarithmic target transformation
* Outlier detection
* K-Fold Cross-Validation
* Early Stopping
* Model Serialization
* Model Diagnostics
* Prediction on unseen data

---

# 🔮 Future Improvements

Possible extensions of the project include:

* hyperparameter optimization using Optuna,
* systematic feature selection,
* SHAP-based model explainability,
* additional ensemble models,
* model stacking / blending,
* automated experiment tracking,
* reproducible training pipelines,
* automated model evaluation,
* deployment through FastAPI or Streamlit,
* Docker containerization,
* CI/CD for automated model testing.

---

# ⚠️ Limitations

The model should be treated as a machine learning prediction system rather than a professional real-estate valuation tool.

House prices are affected by many external factors that may not be represented in the dataset, including:

* local market conditions,
* economic conditions,
* interest rates,
* neighborhood changes,
* supply and demand,
* property-specific characteristics not captured by the dataset.

Therefore, model predictions should be interpreted as **data-driven estimates**, not guaranteed market values.

---

# 👨‍💻 Author

**Paweł Drwal**

Student of Computer Science interested in:

* Machine Learning
* Data Science
* Python
* Data Analysis
* Artificial Intelligence

GitHub:
https://github.com/Drwalpawel20

---

## ⭐ Project Focus

> **From raw housing data to a validated machine learning model capable of generating house-price predictions.**

This project demonstrates the practical application of the complete machine learning workflow rather than focusing only on model training.

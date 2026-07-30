# 🇮🇳 India's Silent Epidemic — Suicide Rate Prediction & Analysis

> **SDG Goal 3: Good Health and Well-Being**
>
> *"When you feel like giving up, just remember the reason why you held on for so long."*

A comprehensive data analysis and machine learning project that explores suicide trends in India (2001–2012), uncovers key contributing factors, and builds a predictive model to classify suicide rate categories. Built as part of an **IBM Internship** by **Team GPG Girls**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Key Findings](#key-findings)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Team](#team)
- [Disclaimer](#disclaimer)

---

## Overview

Suicide is a critical public health issue in India. This project performs an **end-to-end analysis** of suicide data across Indian states and union territories, examining patterns by:

- **Year** (2001–2012 trends)
- **Gender** (Male vs. Female)
- **Age Group** (0–14, 15–29, 30–44, 45–59, 60+)
- **Causes** (Family problems, mental illness, etc.)
- **Professional Profile** (Housewives, farmers, students, etc.)
- **Social Status** (Married, unmarried, widowed, etc.)
- **Means Adopted** (Hanging, poisoning, etc.)

It then trains machine learning models to **predict the category of suicide rate** based on demographic and contextual features.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [Suicides in India 2001–2012](https://www.kaggle.com/datasets) (NCRB data) |
| **File** | `Suicides in India 2001-2012.csv` |
| **Records** | ~237,000 rows |
| **Columns** | `State`, `Year`, `Type_code`, `Type`, `Gender`, `Age_group`, `Total` |

### Data Cleaning Steps

1. Removed aggregated rows (`Total (All India)`, `Total (States)`, `Total (Uts)`)
2. Dropped rows with zero total suicides
3. Removed unspecified/ambiguous cause categories
4. Handled redundant/duplicate category labels

---

## Key Findings

### 📈 Trends Over Time
- A **rapid increase** in the number of suicides from 2001 to 2012.
- The increase among males is significantly greater than among females.

### 👥 Demographics
- **Males** account for a significantly higher proportion of suicides.
- The **15–29 age group** has the highest suicide rate, followed by 30–44.

### 🔍 Top Causes
1. **Family Problems** — leading cause
2. **Insanity / Mental Illness**
3. **Love Affairs**
4. **Dowry Disputes**
5. **Bankruptcy / Sudden Change in Economic Status**

### 💼 Professional Profiles
- **Housewives** have the highest number of suicides, with 44.7% in the 15–29 age group.
- **Farming / Agriculture** is the second-most affected profession.
- Student suicides show a consistent **upward trend** year over year, with 76% in the 15–29 age group.

### 🏛️ State-wise
- **Maharashtra** reports the highest number of suicides.
- Followed by **West Bengal**, **Tamil Nadu**, and **Andhra Pradesh**.

### 💍 Social Status
- **Married individuals** constitute 70.2% of suicide victims.
- Never-married individuals account for 21.9%.

### ⚠️ Means Adopted
- **Hanging** is the most prevalent method.
- **Consuming insecticides/poison** is the second most common.

---

## Machine Learning Pipeline

### Problem Formulation

The `Total` (number of suicides) column is binned into **8 categories**:

| Bin | Range |
|---|---|
| 0-10 | 0 – 10 |
| 10-50 | 10 – 50 |
| 50-100 | 50 – 100 |
| 100-500 | 100 – 500 |
| 500-1000 | 500 – 1,000 |
| 1000-5000 | 1,000 – 5,000 |
| 5000-10000 | 5,000 – 10,000 |
| 10000+ | 10,000+ |

This converts the regression problem into a **multi-class classification** task.

### Preprocessing
- **Label Encoding** for categorical features (`State`, `Type_code`, `Type`, `Gender`, `Age_group`)
- **Feature Engineering**: Created interaction features (`State_Type`, `Gender_Age`)
- **Standard Scaling** on numerical features (`Year`)
- **Train/Test Split**: 70/30

### Models Trained

| Model | Technique | Best Accuracy |
|---|---|---|
| **HistGradientBoostingClassifier** | GridSearchCV (5-fold CV) | ~68% |
| **XGBoost (XGBClassifier)** ✅ | GridSearchCV (5-fold CV) | **Best performing** |

The **XGBoost model** was selected as the final model and saved as `suicide_prediction_model.pkl`.

### Hyperparameter Search Space (XGBoost)
```
learning_rate : [0.01, 0.1, 0.2]
n_estimators  : [100, 200]
max_depth     : [3, 5, 7]
subsample     : [0.8, 1.0]
colsample_bytree : [0.8, 1.0]
```

---

## Web Application

A **Flask web app** is included for making real-time predictions through a simple form interface.

### Input Features

| Field | Description | Example |
|---|---|---|
| State | Indian state or UT | `Maharashtra` |
| Type Code | Category type code | `Causes` |
| Type | Specific type/cause | `Family Problems` |
| Gender | Male / Female | `Male` |
| Age Group | Age range | `15-29` |
| Year | Year (2001–2012) | `2010` |

### Output
A predicted **suicide rate category** (e.g., `100-500`, `1000-5000`).

---

## Project Structure

```
Sucide-prediction/
├── IBM_INTERNSHIP_GPG_GIRLS.ipynb     # Full analysis & model training notebook
├── app.py                              # Flask web application
├── index.html                          # Frontend form template
├── Suicides in India 2001-2012.csv     # Raw dataset
├── suicide_prediction_model.pkl        # Trained XGBoost model (serialized)
├── suicides_in_india_profile_report.html  # Pandas Profiling report
├── requirements.txt                    # Python dependencies
├── download.jpg                        # Project image asset
├── IBM PPT.pdf                         # IBM internship presentation
├── IBM concept note.pdf                # Project concept note
└── README.md                           # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/snehadpatel/Sucide-prediction.git
   cd Sucide-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   # venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

---

## Usage

### Jupyter Notebook (Analysis & Training)
```bash
jupyter notebook IBM_INTERNSHIP_GPG_GIRLS.ipynb
```
Run all cells to reproduce the full analysis pipeline — from data cleaning and EDA through model training and evaluation.

### Web App (Predictions)
```bash
python app.py
```
Fill in the form fields on the web interface and click **Predict** to get the predicted suicide rate category.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3 |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Pandas Profiling |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Web Framework** | Flask |
| **Model Serialization** | Joblib |

---

## Team

**Team GPG Girls** — IBM Internship Project

---

## Disclaimer

> ⚠️ **If you or someone you know is struggling, please reach out for help.**
>
> - **India**: iCall — 9152987821 | Vandrevala Foundation — 1860-2662-345
> - **International**: [findahelpline.com](https://findahelpline.com)
>
> This project is intended for **educational and research purposes only**. The predictions made by this model should not be used as a substitute for professional mental health assessment.

---

## License

This project is open-source and available for educational use.
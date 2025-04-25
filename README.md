

# Credit Risk Prediction Application
## Problem Statement
Background: Financial institutions face significant challenges in assessing the creditworthiness of loan applicants. Accurate credit risk prediction is crucial for minimizing defaults and ensuring the stability of the lending system. The German Credit dataset provides a comprehensive set of features related to applicants' financial history, personal information, and loan details, making it an ideal resource for developing predictive models.
Objective: Develop a machine learning model to predict the credit risk of loan applicants using the **German Credit dataset**. The model should classify applicants into two categories: good credit risk and bad credit risk. Additionally, provide insights into the key factors influencing credit risk and suggest strategies for improving the credit evaluation process.


## Dataset Overview

The dataset consists of financial history, personal information, and loan details for applicants.  
**Key columns include:**
- `Age`
- `Sex`
- `Job`
- `Housing`
- `Saving Account`
- `Checking Account`
- `Credit Amount`
- `Duration`
- `Purpose`
- `Risk` (Target variable: Good / Bad)

**Sample Data:**

| Age | Sex   | Job | Housing | Saving Account | Checking Account | Credit Amount | Duration | Purpose  | Risk |
|:-----|:--------|:-----|:------------|:----------------|:-----------------|:----------------|:------------|:-------------|:------|
| 61  | male  | 1   | own       | rich              | unknown            | 3059             | 12         | radio/TV   | Good  |
| 44  | male  | 2   | rent      | Quite rich        | little             | 2647             | 6          | radio/TV   | Good  |
| 25  | female| 2   | rent      | little            | moderate           | 1295             | 12         | car        | Bad   |

---

##  Deliverables

### 1 Data Exploration and Preprocessing
- Handled null values by replacing them with **"unknown"**.
- Used **Label Encoding** for categorical columns (`Sex`, `Job`, etc.)
- Applied **StandardScaler** to numerical columns (`Age`, `Credit Amount`, etc.)
- Determined credit amount and duration thresholds to classify `Risk`.

---

### 2 Model Development
- Split the dataset into **features (X)** and **target (Y)**.
- Developed and evaluated two models:
  - **Random Forest Classifier**
  - **Support Vector Machine (SVM)** (with hyperparameter tuning)
- Evaluated both models using a **Classification Report**.

---

### 3 Model Interpretation and Insights
- Plotted **feature importance** for both Random Forest and SVM models to identify key risk factors.

---

The first 3 deliverables are found in the jupyter notebook. Out of the two models, SVM is selected for UI.

### 4 Streamlit Web Application
- Created an interactive **Streamlit web app**.
- Allowed users to:
  - **Upload a credit dataset**
  - Enter new applicant data via form inputs
  - Predict applicant’s credit risk using the trained **SVM model**
  - Display prediction result and model confidence


## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/credit-risk-predictor.git
   cd credit-risk-predictor
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Results

- Achieved high classification accuracy using both Random Forest and SVM.
- Identified key influencing factors:
  - **Credit Amount**
  - **Duration**


## Demo

[BFSI_report.docx](https://github.com/user-attachments/files/19906968/BFSI_report.docx)
https://drive.google.com/file/d/1EK_0-VHydfx2wJR7wDvlOJobj7X_dUX1/view?usp=sharing



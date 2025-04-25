
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("Credit Risk Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload German Credit Data CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Basic cleaning
    if 'Unnamed: 0' in data.columns:
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # Handle missing values
    st.write("Missing values in each column:")
    st.write(data.isnull().sum())

    data['Saving accounts'].fillna('unknown', inplace=True)
    data['Checking account'].fillna('unknown', inplace=True)

    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    data_encoded = data.copy()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        label_encoders[col] = le  # Save encoders for later interpretation

    scaler = StandardScaler()
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])


    credit_threshold = data_encoded['Credit amount'].median()
    duration_threshold = data_encoded['Duration'].median()

    data_encoded['Risk'] = (
        (data_encoded['Credit amount'] > credit_threshold) &
        (data_encoded['Duration'] < duration_threshold)
    ).astype(int)

    # data_encoded['Risk'].value_counts()

    # Split data
    X = data_encoded.drop(columns=['Risk'])
    y = data_encoded['Risk']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

    # Train SVM with GridSearchCV
    st.write("Training SVM model...")
    svm = SVC(probability=True, random_state=42)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_svm = grid_search.best_estimator_

    st.success(f"Best SVM Parameters: {grid_search.best_params_}")

    # Show classification report
    y_pred = best_svm.predict(X_test)
    st.subheader("Model Performance on Test Set")
    st.text(classification_report(y_test, y_pred))

    # Feature Importance
    st.subheader("Feature Importance (Permutation Importance)")
    result = permutation_importance(best_svm, X_test, y_test, n_repeats=10, random_state=42)
    importances = result.importances_mean
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    st.pyplot(fig)

    # New Prediction Input
    st.subheader("Predict Risk for New Applicant")
    new_input = {}

    # Categorical Inputs (typed in by user)
    for col in categorical_cols:
        new_input[col] = st.text_input(f"Enter {col}")

    # Numerical Inputs
    for col in numerical_cols:
        new_input[col] = st.number_input(f"Enter {col}")

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([new_input])

        # Apply Label Encoding
        for col in categorical_cols:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Apply Scaling
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Ensure columns order matches X_train
        input_df = input_df[X.columns]

        prediction = best_svm.predict(input_df)[0]
        probability = best_svm.predict_proba(input_df)[0][prediction]
        result_text = "Good" if prediction == 1 else "Bad"
        st.success(f"Predicted Risk: {result_text} (Confidence: {probability:.2f})")

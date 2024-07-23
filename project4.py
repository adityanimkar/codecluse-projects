import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("heart.csv")

# Preprocess data
X = data.drop(columns=['target'])
y = data['target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Streamlit app
st.title("Heart Disease Risk Prediction")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

# Prediction button
if st.button("Predict"):
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = model.predict(input_data)
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.write(f"Predicted Risk: {risk}")

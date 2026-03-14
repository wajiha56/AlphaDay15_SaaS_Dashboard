import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Background AI Training
data = {
    'Age': [22, 25, 45, 38, 50, 23, 30, 35, 40, 28, 21, 42, 55, 26, 31],
    'Minutes_on_Site': [5, 15, 2, 20, 25, 1, 10, 18, 8, 30, 3, 22, 12, 19, 28],
    'Bought_Product': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)
X = df[['Age', 'Minutes_on_Site']]
y = df['Bought_Product']

model = LogisticRegression()
model.fit(X.values, y)

# 2. Web Dashboard UI
st.title("📊 Customer Prediction SaaS")
st.write("Welcome to the Interactive AI Dashboard. Enter customer details to predict if they will buy.")

# User Inputs
st.sidebar.header("Enter Customer Data")
age = st.sidebar.slider("Customer Age", 18, 70, 30)
minutes = st.sidebar.slider("Minutes on Site", 1, 60, 15)

st.write(f"**Analyzing Customer:** {age} years old, spent {minutes} minutes on site.")

# 3. Prediction Button
if st.button("Predict Buying Behavior"):
    prediction = model.predict([[age, minutes]])
    
    st.subheader("AI Prediction Result:")
    if prediction[0] == 1:
        st.success("✅ The AI predicts: THIS CUSTOMER WILL BUY.")
        st.balloons()
    else:
        st.error("❌ The AI predicts: THIS CUSTOMER WILL NOT BUY.")

st.write("---")
st.write("Built by Wajiha Arshad | AI & SaaS Developer")
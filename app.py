import streamlit as st
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Page Layout Title
st.title("💼 Simple Salary Predictor")
st.write("Move the sliders below to calculate an estimated salary!")
st.markdown("---")

# 2. Train the AI Model (Using ONLY Experience and Skills now)
@st.cache_data
def train_brain():
    df = pd.read_csv('job_salary_prediction_dataset.csv')
    
    # We only take TWO columns here to match your goal
    X = df[['experience_years', 'skills_count']]
    y = df['salary']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

ai_model = train_brain()

# 3. Create interactive sliders for the website users
years = st.slider("How many years of experience?", 0, 20, 5)
skills = st.slider("How many tech skills do you have?", 0, 30, 10)

st.markdown("---")

# 4. Predict the salary matching the two inputs perfectly
if st.button("🔮 Calculate Salary", type="primary"):
    # We pass exactly TWO variables here to match the training data
    prediction = ai_model.predict([[years, skills]])[0]
    st.success(f"### Estimated Salary: ${prediction:,.2f} per year")
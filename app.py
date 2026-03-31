import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt

#Dataset
data = pd.read_csv("student_data.csv")

#Features
X = data[['study_hours','sleep_hours','exercise_hours','screen_time']]

#Targets
y_marks = data['marks']
y_health = data['health_risk']

#Models
reg_model = LinearRegression().fit(X,y_marks)
clf_model = LogisticRegression().fit(X,y_health)

#Title

st.title("Student Health & Performance Predictor")
st.write("Enter your daily habits:")

#Inputs
study = st.slider("Study Hours",0,10,5)
sleep = st.slider("Sleep Hours",0,10,7)
exercise = st.slider("Exercise Hours",0,5,1)
screen = st.slider("Screen Time",0,10,3)

#Prediction Button
if st.button("Predict"):
    input_data = [[study,sleep,exercise,screen]]
    marks = reg_model.predict(input_data)[0]
    health = clf_model.predict(input_data)[0]

    st.subheader("Results")
    st.write(f"📊 Predicted Marks: {marks:.2f}")

    if health == 1:
        st.error("⚠️ Health Risk: HIGH")
    else:
        st.success("✅ Health Risk: LOW")

    #Graph
    st.subheader("📈 Performance Trend")
    fig,ax = plt.subplots()
    ax.plot(data['marks'])
    ax.set_xlabel("Students")
    ax.set_ylabel("Marks")
    st.pyplot(fig)
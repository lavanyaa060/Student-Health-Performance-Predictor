import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt

#Dataset
data = pd.read_csv("student_data.csv")

#Input
X = data[['study_hours','sleep_hours','exercise_hours','screen_time']]

#Output
y_marks = data['marks']
y_health = data['health_risk']

#Models
reg_model = LinearRegression()
clf_model = LogisticRegression()

#Training Models
reg_model.fit(X,y_marks)
clf_model.fit(X,y_health)

#User Input
study = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))
exercise = float(input("Enter exercise hours: "))
screen = float(input("Enter screen time: "))

input_data = [[study,sleep,exercise,screen]]

#Predictions
predicted_marks = reg_model.predict(input_data)
health_status = clf_model.predict(input_data)

#Results
print("\nRESULTS")
print(f"Predicted Marks:{predicted_marks[0]:.2f}")

if health_status[0] == 1:
   print("Health Risk:HIGH⚠️")
else:
   print("Health Risk:LOW✅")

#Forecasting(simple graph)
plt.plot(data['marks'])
plt.title("Performance Trend")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()
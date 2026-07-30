from sklearn.linear_model import LinearRegression

# Training Data
X = [
    [1,15,60],
    [2,15,65],
    [3,16,70],
    [4,16,75],
    [5,17,80],
    [6,17,82],
    [7,18,85],
    [8,18,88],
    [9,18,90],
    [10,19,94]
]

y = [3.5,4.2,5.1,6.0,7.2,8.0,9.1,10.0,11.2,12.5]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Prediction
new_employee = [[7,18,86]]
prediction = model.predict(new_employee)

print("Predicted Salary (Lakhs):", prediction[0])

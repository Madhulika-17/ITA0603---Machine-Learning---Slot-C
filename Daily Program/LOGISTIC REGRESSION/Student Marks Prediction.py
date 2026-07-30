from sklearn.linear_model import LinearRegression

# Training Data
X = [
    [2,70,60],
    [3,75,65],
    [4,80,70],
    [5,82,75],
    [6,85,80],
    [7,88,85],
    [8,90,90],
    [9,92,94],
    [10,95,96],
    [11,97,98]
]

y = [55,60,68,74,80,86,91,95,98,100]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Prediction
new_student = [[7,90,88]]
prediction = model.predict(new_student)

print("Predicted Final Marks:", prediction[0])

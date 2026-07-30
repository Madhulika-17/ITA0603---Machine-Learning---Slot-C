from sklearn.linear_model import LinearRegression

# Training Data
X = [
    [800,2,10],
    [1000,2,8],
    [1200,3,6],
    [1400,3,5],
    [1600,4,4],
    [1800,4,3],
    [2000,5,2],
    [2200,5,1],
    [2400,6,1],
    [2600,6,0]
]

y = [35,45,55,65,75,85,95,105,115,125]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Prediction
new_house = [[1700,4,3]]
prediction = model.predict(new_house)

print("Predicted House Price (Lakhs):", prediction[0])

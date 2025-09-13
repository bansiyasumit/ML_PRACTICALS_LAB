import matplotlib.pyplot as plt

# Input data
X = [0, 1, 2, 3, 3, 5, 5, 5, 6, 7, 7, 10]
Y = [96, 85, 82, 74, 95, 68, 76, 84, 58, 65, 75, 50]

# Calculate means
mean_x = sum(X) / len(X)
mean_y = sum(Y) / len(Y)
print(f"Mean of X: {mean_x}")
print(f"Mean of Y: {mean_y}")

# Calculate slope (m)
numerator = sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(len(X)))
denominator = sum((X[i] - mean_x)**2 for i in range(len(X)))
m = numerator / denominator
print(f"Slope (m): {m}")

# Calculate intercept (c)
c = mean_y - m * mean_x
print(f"Y-intercept (c): {c}")

# Define prediction function
def predict_y(x, m, c):
    return m * x + c

# Interactive prediction
try:
    user_x = float(input("Enter an X value to predict Y: "))
    predicted_y_for_user_x = predict_y(x=user_x, m=m, c=c)
    print(f"Predicted Y value for X={user_x}: {predicted_y_for_user_x}")
except ValueError:
    print("Invalid input. Please enter a number.")
    exit()

# Visualization
regression_line = [predict_y(x, m, c) for x in X]

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Actual Data', marker='o')
plt.plot(X, regression_line, color='red', label='Regression Line')
plt.scatter(user_x, predicted_y_for_user_x, color='green', label=f'Predicted (X={user_x})', marker='x', s=100)
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Linear Regression: Hours vs Marks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

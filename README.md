# Linear Regression Model for Predicting Student Marks

This project implements a simple linear regression model to predict student marks based on the number of hours they study. The model is built from scratch using basic Python calculations for the mean, slope, and y-intercept.

## Input Data

The input data consists of two sets of values:

- **X (Independent Variable):** Hours studied. The provided values are: {0, 1, 2, 3, 3, 5, 5, 5, 6, 7, 7, 10}
- **Y (Dependent Variable):** Marks obtained. The corresponding values are: {96, 85, 82, 74, 95, 68, 76, 84, 58, 65, 75, 50}

## Formulas Used

The linear regression model follows the equation: **Y = mx + c**, where:

- **Y** is the predicted dependent variable (Marks)
- **x** is the independent variable (Hours)
- **m** is the slope of the regression line
- **c** is the y-intercept

The calculations for the slope and intercept are based on the following formulas:

### Mean of X and Y

The mean of a set of values is calculated as the sum of the values divided by the number of values.

$\text{mean}(X) = \frac{\Sigma X}{n}$

$\text{mean}(Y) = \frac{\Sigma Y}{n}$

### Slope (m)

The slope of the regression line is calculated using the formula:

$m = \frac{\Sigma[(x_i - \text{mean}(X))(y_i - \text{mean}(Y))]}{\Sigma(x_i - \text{mean}(X))^2}$

### Y-intercept (c)

The y-intercept is calculated using the formula derived from the linear equation Y = mx + c:

$c = \text{mean}(Y) - m * \text{mean}(X)$

## How to Run the Code

1. **Open the Notebook:** Open the provided Python notebook file (e.g., `linear_regression_model.ipynb`) in a Jupyter Notebook environment or Google Colab.
2. **Run Cells:** Execute each code cell sequentially. The notebook is structured to define the data, calculate the means, slope, and intercept, and define a prediction function.

## Making Predictions

After running the notebook, you can use the `predict_y` function to predict the marks for a new number of hours studied.

The function takes the following arguments:

- `x`: The new value for hours studied.
- `m`: The calculated slope of the regression line.
- `c`: The calculated y-intercept.

To make a prediction, simply call the function with the desired hours and the calculated `m` and `c` values from the notebook execution.

```python
# Example: Predict marks for 9 hours of study
predicted_marks = predict_y(x=9, m=m, c=c)
print(f"Predicted marks for 9 hours of study: {predicted_marks}")
```

The notebook also includes an interactive section where you can input an X value and get a prediction directly.

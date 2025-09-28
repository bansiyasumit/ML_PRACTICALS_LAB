# ML_PRACTICALS_LAB

## 📘 Assignment 1: Linear Regression – Predicting Student Marks

This project builds a simple linear regression model from scratch to predict student marks based on hours studied.

### 🔹 File: `assignment1.py`

### 🔹 Input
- **X (Hours):** [0, 1, 2, 3, 3, 5, 5, 5, 6, 7, 7, 10]
- **Y (Marks):** [96, 85, 82, 74, 95, 68, 76, 84, 58, 65, 75, 50]

### 🔹 Method
- Calculates mean, slope (m), and intercept (c)
- Predicts using: `Y = mx + c`

### 📈 Visualization
![Linear Regression Output](output_LR.png)

---



---

# 📝 LAB 2: Logistic Regression for Insurance Prediction

This project uses **Logistic Regression** (implemented via `scikit-learn`) to predict whether an individual will purchase insurance based on their **age**.

## 💾 Project Files

* `LOGISTIC_REGRESSION_LAB_2.ipynb`: The Jupyter Notebook containing all the code for data loading, visualization, model training, evaluation, and prediction.
* `insurance_data_logistic`: The dataset used for training the model.

## 🚀 Model Details

### 1. Training & Features
* **Feature (X):** `age`
* **Target (y):** `bought_insurance` (0 = No, 1 = Yes)
* **Training Method:** Train-Test Split (80/20) and 5-Fold Cross-Validation.

### 2. Model Parameters
The trained model uses these parameters for the logistic function:
* **Weight ($\beta_1$ for Age):** $0.1201$
* **Bias ($\beta_0$ / Intercept):** $-4.8564$

$$\text{Probability} = \frac{1}{1 + e^{-(\mathbf{0.1201} \cdot \text{Age} - \mathbf{4.8564})}}$$

### 3. Evaluation Metrics
| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | $\approx 83.33\%$ |
| **Average 5-Fold CV Accuracy** | $\mathbf{86.00\%}$ |

## 📊 Sigmoid Curve

The output of the notebook includes a visualization of the **Sigmoid Curve**, showing the increasing probability of buying insurance as age increases.

* Ages in the lower range (e.g., 20) have a low probability ($\approx 8\%$).
* Ages in the higher range (e.g., 60) have a high probability ($\approx 91\%$).


# 📝 Assignment: LAB 3 - LOGISTIC REGRESSION FROM SCRATCH

This project implements **Logistic Regression** entirely from **scratch** using NumPy, bypassing external libraries like `scikit-learn` to demonstrate the core mathematical model.

---

## 💻 Implementation

* **Model:** `LogisticRegressionScratch` class implementing the Sigmoid function, Binary Cross-Entropy Cost, and Gradient Descent.
* **Data:** `insurance_data_logistic` (Age $\rightarrow$ Bought Insurance 0/1).
* **Preprocessing:** The **Age** feature is **standardized** to ensure optimal convergence of the gradient descent algorithm.

---

## 📊 Results

| Metric | Value |
| :--- | :--- |
| **Learned Weights ($\beta$ for Scaled Age)** | $\mathbf{2.0971}$ |
| **Learned Bias ($\beta_0$ / Intercept)** | $\mathbf{0.1107}$ |
| **Training Accuracy** | $\approx 88.89\%$ |

The notebook includes plots demonstrating the decrease in the **Cost Function** over iterations and the final **Sigmoid Curve** fitted to the data.

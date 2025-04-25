import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

# Problem 1: Diabetes

data = load_diabetes(as_frame=True)
df = data['frame']

print(df.head())
print(data.DESCR)

plt.hist(df['target'], bins=25)
plt.xlabel("Target")
plt.title("Target Distribution")
plt.show()

sns.heatmap(df.corr().round(2), annot=True)
plt.title("Correlation Matrix")
plt.show()

plt.scatter(df['bmi'], df['target'], label='bmi')
plt.scatter(df['s5'], df['target'], label='s5', alpha=0.5)
plt.legend()
plt.title("BMI and S5 vs Target")
plt.show()

X_base = df[['bmi', 's5']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(y_test, y_pred))
r2_base = r2_score(y_test, y_pred)
print("Base model RMSE:", rmse_base, "R2:", r2_base)

"""
I added 'bp' (blood pressure) because it's medically relevant and moderately correlated with the target.
"""

X_bp = df[['bmi', 's5', 'bp']]
X_train, X_test, y_train, y_test = train_test_split(X_bp, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse_bp = np.sqrt(mean_squared_error(y_test, y_pred))
r2_bp = r2_score(y_test, y_pred)
print("With bp RMSE:", rmse_bp, "R2:", r2_bp)

"""
Adding 'bp' slightly improved model performance.
"""

X_all = df.drop(columns="target")
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse_all = np.sqrt(mean_squared_error(y_test, y_pred))
r2_all = r2_score(y_test, y_pred)
print("All features RMSE:", rmse_all, "R2:", r2_all)

"""
Using all variables gave the best results with highest R2 and lowest RMSE.
"""

# Problem 2: Profit Prediction

df = pd.read_csv("50_Startups.csv")
print(df.head())

"""
The dataset includes R&D Spend, Administration, Marketing Spend, State, and Profit.
"""

df_encoded = pd.get_dummies(df, drop_first=True)
sns.heatmap(df_encoded.corr().round(2), annot=True)
plt.title("Correlation Matrix")
plt.show()

"""
R&D Spend and Marketing Spend show strongest correlation with Profit.
"""

plt.scatter(df['R&D Spend'], df['Profit'])
plt.title("R&D Spend vs Profit")
plt.show()

plt.scatter(df['Marketing Spend'], df['Profit'])
plt.title("Marketing Spend vs Profit")
plt.show()

X = df[['R&D Spend', 'Marketing Spend']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("Train RMSE:", rmse_train, "R2:", r2_train)
print("Test RMSE:", rmse_test, "R2:", r2_test)

"""
R&D and Marketing Spend are good predictors due to strong linear relationship with Profit.
"""

# Problem 3: Car MPG Prediction

auto = pd.read_csv("Auto.csv", na_values="?").dropna()
print(auto.head())

X = auto.drop(columns=["mpg", "name", "origin"])
y = auto["mpg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = np.logspace(-3, 2, 50)
r2_ridge = []
r2_lasso = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    r2_ridge.append(ridge.score(X_test, y_test))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    r2_lasso.append(lasso.score(X_test, y_test))

plt.plot(alphas, r2_ridge, label='Ridge')
plt.plot(alphas, r2_lasso, label='Lasso')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('R2 vs Alpha')
plt.legend()
plt.show()

best_alpha_ridge = alphas[np.argmax(r2_ridge)]
best_alpha_lasso = alphas[np.argmax(r2_lasso)]

print("Best Ridge alpha:", best_alpha_ridge, "R2:", max(r2_ridge))
print("Best Lasso alpha:", best_alpha_lasso, "R2:", max(r2_lasso))

"""
The optimal alpha is chosen based on best R2 score.
Ridge and Lasso both help improve generalization by reducing overfitting.
"""

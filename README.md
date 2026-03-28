# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and prepare data – Import dataset, select predictor variables (enginesize, horsepower, citympg, highwaympg) and target (price).
2. Split dataset – Divide into training and testing sets using train_test_split.
3. Scale features – Standardize predictors with StandardScaler for better regression performance.
4.Train model & predict – Fit LinearRegression on training data, generate predictions on test data.
5. Evaluate & visualize – Compute metrics (MSE, RMSE, R², MAE), check residuals (Durbin‑Watson, homoscedasticity), and plot actual vs predicted prices. 

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Harish Pranav
RegisterNumber:  212225040117
*/

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
df=pd.read_csv('CarPrice_Assignment.csv')
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
print("Name: Harish Pranav")
print("Reg no: 212225040117")
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}:{model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")
mse=mean_squared_error(y_test, y_pred)
print('Mean squared error=',mse)
rmse=np.sqrt(mse)
print('Root mean squared error=',rmse)
r2score=r2_score(y_test, y_pred)
print('R-squared=',r2score)

# 1. Linearity check
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity check: Actual vs Predicted prices")
plt.xlabel("Actual Price($)")
plt.ylabel("Predicted Price($)")
plt.grid(True)
plt.show()

# 2. Independence (Durbin-Watson)
residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic:{dw_test:.2f}","\n(Values close to 2 indicate no autocorrelation)")

# 3. Homoscedasticity
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

# 4. Normality of residuals
fig, (ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()


```

## Output:
<img width="725" height="359" alt="image" src="https://github.com/user-attachments/assets/064d1a4e-df7c-4e61-85ab-8b8f3b04958a" />
<img width="1467" height="682" alt="image" src="https://github.com/user-attachments/assets/7f917feb-ecd9-4269-8c81-6a2af83c1a5d" />
<img width="758" height="76" alt="image" src="https://github.com/user-attachments/assets/0fda5f4a-1868-4267-b3ed-c8ae8a18d4a3" />
<img width="1485" height="686" alt="image" src="https://github.com/user-attachments/assets/a94aa797-38f0-4cd7-afcb-f9beba8eabc2" />
<img width="1524" height="579" alt="image" src="https://github.com/user-attachments/assets/ca99dc39-0fa4-4321-95f1-5da82d1403fb" />








## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.

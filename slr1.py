import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=1)

#Training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_train)

#Graphing Regression
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, regressor.predict(x_train), color='red')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience (Year)')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, regressor.predict(x_test), color='red')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

print("Coefficient:", regressor.coef_)
print("Constant:", regressor.intercept_)
#Equation: ŷ = 27417.513 + 8829.2714x

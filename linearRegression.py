import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, learnRate, iterationNo):
        self.learnRate = learnRate
        self.iterationNo = iterationNo

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.m, self.n = x.shape

        self.weight = np.zeros(self.n) # A dataset can have any no of features and each feature will have a slope associated with it
        self.bias = 0 # A dataset will contain only one y intercept value

        # Gradient Descent Algorithm
        for i in range(self.iterationNo):
            self.update()

    def update(self):
        yPrediction = self.predict(self.x)

        dweight = - (2 * (self.x.T).dot(self.y - yPrediction)) / self.m
        dbias = - 2 * np.sum(self.y - yPrediction) / self.m

        self.weight = self.weight - self.learnRate * dweight
        self.bias = self.bias - self.learnRate * dbias

    def predict(self, x):
        return x.dot(self.weight) + self.bias


df = pd.read_csv('salary_data.csv')

print(df.head())
print(df.isnull().sum())

x = df.drop(columns = 'Salary', axis = 1)
y = df['Salary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)

model = LinearRegression(learnRate = 0.017, iterationNo = 1000)
model.fit(x_train, y_train)

print("Weight : ", model.weight[0])
print("Bias : ", model.bias)

prediction = model.predict(x_test)

plt.scatter(x_test, y_test, color = 'r')
plt.plot(x_test, prediction, color = 'b')
plt.show()
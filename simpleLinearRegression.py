# Import the necessary libraries
import numpy
import matplotlib.pyplot as plot
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the dataset
dataset = pandas.read_csv('salaryData.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the dataset into the training set and test set
# We're splitting the data in 1/3, so out of 30 rows, 20 rows will go into the training set,
# and 10 rows will go into the testing set.
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Creating a LinearRegression object and fitting it
# on our trainging set.
linearRegressor = LinearRegression()
linearRegressor.fit(xTrain, yTrain)

# Predicting the test set results
yPrediction = linearRegressor.predict(xTest)

# Visualising the training set results
plot.scatter(xTrain, yTrain, color = 'red')
plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
plot.title('Salary vs Experience (Training set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

# Visualising the test set results
plot.scatter(xTest, yTest, color = 'red')
plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
plot.title('Salary vs Experience (Test set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()
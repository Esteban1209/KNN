# Pandas will allow us to create a dataframe of the data so it can be used and manipulated
import pandas as pd
# Regression Tree Algorithm
from sklearn.tree import DecisionTreeRegressor
# Split our data into a training and testing data
from sklearn.model_selection import train_test_split
from pandas import DataFrame as df


data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv")
#CRIM: Crime per capita

#ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.

#INDUS: Proportion of non-retail business acres per town

#CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

#NOX: Nitric oxides concentration (parts per 10 million)

#RM: Average number of rooms per dwelling

#AGE: Proportion of owner-occupied units built prior to 1940

#DIS: Weighted distances to ﬁve Boston employment centers

#RAD: Index of accessibility to radial highways

#TAX: Full-value property-tax rate per $10,000

#PTRAIO: Pupil-teacher ratio by town

#LSTAT: Percent lower status of the population

#MEDV: Median value of owner-occupied homes in $1000s

data.head()
data.shape
data.isna().sum()
data.dropna(inplace=True)
data.isna().sum()
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]
X.head()
Y.head()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

regression_tree = DecisionTreeRegressor(criterion = 'mse')
regression_tree.fit(X_train, Y_train)
regression_tree.score(X_test, Y_test)

prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)

regression_tree = DecisionTreeRegressor(criterion = "mae")

regression_tree.fit(X_train, Y_train)

print(regression_tree.score(X_test, Y_test))

prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)

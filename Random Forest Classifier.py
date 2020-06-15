# In this project we want to predict based on a given data set if a loan will be repaid or not. 
# The dependant variable is 'not.fully.paid'
# The code needs to be run with Jupyter Notebook

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Loading the data
loans = pd.read_csv('loan_data.csv')

# Displaying aggregated information about the data
loans.info()
print('\n')
loans.describe()
print('\n')
loans.head()

# Exploratory Data Analysis

# Creating a histogram of the FICO score showing each type of Credit Policy (1 or 0).
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')

# Making a countplot for each loan purpose showing how many of them have been paid of not
plt.figure(figsize = (12,8))
sns.countplot(x = 'purpose', data = loans, hue = 'not.fully.paid')

# Creating a joint plot to show the correlation between the fico score and the interest rate
sns.jointplot('fico', 'int.rate', data = loans)

# Getting the data ready for the model:

# Tranforming the Purpose categorical column into a dummy variable
cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns = cat_feats, drop_first=True)

# Train Test Split the data
X = final_data.drop('not.fully.paid', axis = 1)
y = final_data['not.fully.paid']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Training the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier(n_estimators=200)
forest_classifier.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = forest_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))




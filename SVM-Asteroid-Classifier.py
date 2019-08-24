"""

Using SVMs to classify asteroids as harmful or not

Author : Pranay Venkatesh

NOTE : All date and time measurements used for processing are Epoch Date and Time measurement, which is a convenient way to deal with time.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

"""

Selecting features which are import for classification.

Certain parameters are obviously NOT important such as the name of the asteroid and it's reference ID.

Unimportant parameters:
Name of the asteroid
Neo Reference ID
Orbiting planet (since it's 'Earth' for everything)
Close approach date (Since the date is given in Unix Time as 'Epoch Date Close Approach')
Equinox (Since it's J2000 for everything)
Same quantities in other units (If a given quantity is given in km, we'll exclude the metres, miles and other quantities)


For parameters, we're unsure of, we'll use correlation to figure out the important parameters


"""

df = pd.read_csv("nasa.csv")

# Converting a date to seconds format
df ['Orbit Determination Date'] = pd.to_datetime(df['Orbit Determination Date'])
for ele in df['Orbit Determination Date']:
    ele = (ele - datetime(1970, 1, 1)).total_seconds()

for i in range (len(df['Orbit Determination Date'])):
    df['Orbit Determination Date'][i] = (df['Orbit Determination Date'][i] - datetime(1970, 1, 1)).total_seconds()

print (df['Orbit Determination Date'].head())


# Understanding the type of each data element.
print(df.dtypes)

# First we'll drop the unimportant parameters
df = df.drop(['Neo Reference ID', 'Name', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Close Approach Date', 'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)', 'Miss Dist.(kilometers)', 'Miss Dist.(miles)', 'Orbiting Body', 'Equinox'], axis=1)


# Using correlation to figure out the other important parameters
corrmat = df.corr()
plt.figure(figsize = (20,20))
g = sns.heatmap(df[corrmat.index].corr(), annot=True, cmap="RdYlGn")    # Heatmap to show correlation
plt.show()

# Only taking elements with good correlation
corrmat = abs(corrmat['Hazardous'])
relev_features = corrmat[corrmat > 0.05]
df = df[relev_features.index]

##def extract_features(data):
##    fts = data.drop(['Hazardous'], axis=1)
##    labels = data['Hazardous']
##    trainmat = []
##    trainlab = []
##    testmat = []
##    testlab = []
##    for i in range(len(fts)):
##        if (random.uniform(0,1) > 0.9):
##            trainmat.append(fts.iloc[i])
##            trainlab.append(labels.iloc[i])
##        else:
##            testlab.append(fts.iloc[i])
##            trainlab.append(labels.iloc[i])
##    trainmat = pd.DataFrame(trainmat)
##    trainlab = pd.DataFrame(trainlab)
##    testmat = pd.DataFrame(testmat)
##    testlab = pd.DataFrame(testlab)
##    return trainmat, trainlab, testmat, testlab
#features_matrix, labels, test_matrix, test_labels = extract_features(df)

# Settin up the SVC

features_matrix, test_matrix, labels, test_labels = train_test_split(df, df['Hazardous'], test_size=0.2, random_state=0)    # Splitting the data so that 20% goes for testing while 80% goes for training
print(features_matrix.head())
print(test_matrix.head())
model = svm.SVC()
model.fit(features_matrix, labels)

# Testing the model

predicted_labels = model.predict(test_matrix)
print("TESTING ACCURACY=")
print(accuracy_score(test_labels, predicted_labels))

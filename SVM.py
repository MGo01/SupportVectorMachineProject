from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
from sklearn import metrics
import scipy.stats
import numpy as np
import pandas as pd
import math

# Returns a dictionary that stores the class number as a key
# and the class values as its value.
def divideClasses(dataset):
  classDict = dict()
  divideByClass = dataset.groupby('glass_type')

  for groups, data in divideByClass:
    classDict[groups] = data

  return classDict

# Returns a list of class priors for each respective
# class based on index
def calculateClassPriors(classesDict, numOfClasses):
  priorsList = list()
  
  for groups in classesDict:
    sizeOfGroup = len(classesDict[groups]) 
    priorsList.append(sizeOfGroup / numOfClasses)

  return priorsList

def entropyCalculation(classPriorList):
  sum = 0

  for i in range(len(classPriorList)):
    sum += -classPriorList[i] * math.log(classPriorList[i], 10)
  
  return sum
  
# Returns a dictionary that stores class number as its keys
# and statistics regarding the class data.
def createStatsDict(dataset):
  newDict = dict()
  divideByClass = dataset.groupby('glass_type')

  for groups, data in divideByClass:
    meanSeries = data[['refractive_index','Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron']].mean()
    stdSeries = data[['refractive_index','Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron']].std()

    df = pd.concat([meanSeries, stdSeries], axis=1)
    df.columns= ["Mean","Standard Deviation"]
    newDict[groups] = df

  return newDict
  


def gaussianpdf(x, mean, sigma):
  exponent = (-1/2) * ((x - mean) / sigma)**2
  numerator = math.exp(exponent)
  denominator = sigma * (2 * math.pi)**0.5

  return float(numerator / denominator)

# Get random split of data to work on.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8290)

# Create and fit the Pre-computed weighted SVC 
# for training and testing.
pcWeight_clf = svm.SVC(C=5, kernel='precomputed', gamma='auto', class_weight={5: 10, 6: 12, 7: 8})

gram_train = np.dot(x_train, x_train.T)
pcWeight_clf.fit(gram_train, y_train)
gram_test = np.dot(x_test, x_train.T)

y_predTrain = pcWeight_clf.predict(gram_train)
trainingAccuracy = accuracy_score(y_train, y_predTrain)
print("Weighted PC SVC Training Accuracy: {}".format(trainingAccuracy))

y_predTest = pcWeight_clf.predict(gram_test)
testAccuracy = accuracy_score(y_test, y_predTest)
print("Weighted PC SVC Testing Accuracy: {}".format(testAccuracy))

print()

# Create and fit the linear SVC for training
# and testing with the same data splits from the train_split()
linear_clf = svm.SVC(C=5, kernel='linear', gamma='auto')
linear_clf.fit(x_train, y_train)

y_predTrain = linear_clf.predict(x_train)
trainingAccuracy = accuracy_score(y_train, y_predTrain)
print("Linear SVC Training Accuracy: {}".format(trainingAccuracy))

y_predTest = linear_clf.predict(x_test)
testAccuracy = accuracy_score(y_test, y_predTest)
print("Linear SVC Testing Accuracy: {}".format(testAccuracy))

statsDict = createStatsDict(glass_dataset)

for groups in statsDict:
  print("Class: {}".format(groups))
  print(statsDict[groups])
  print()

x = glass_dataset.drop(['glass_type'], axis = 1)
y = glass_dataset['glass_type'].values

# Modeling and fitting the data to a Gaussian 
# Naive Bayes classifier.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

iter = 0
temp = list()
classConditionals = list()

for group in statsDict:
  temp.append(gnb.class_prior_[iter] * scipy.stats.norm.pdf((statsDict[group]['Mean'], statsDict[group]['Standard Deviation'])))
  classConditionals.append(np.max(temp))
  iter += 1

classes = list()

for i in range(8):
  if (i == 0 or i == 4):
    continue;
  else:
    classes.append(i)
 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
new_columns = ['id_number','refractive_index','Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron', 'glass_type']
glass_dataset = pd.read_csv(url, names = new_columns, header = None, skiprows = 0, usecols=['refractive_index','Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron', 'glass_type'])

glass_dataset = glass_dataset.sample(frac = 1, random_state = 123).reset_index(drop = True)
newDict = divideClasses(glass_dataset)
newList = list()
numOfClasses = 214;

newList = calculateClassPriors(newDict, numOfClasses)


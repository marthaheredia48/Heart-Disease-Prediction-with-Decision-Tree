import pandas as pd

import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows

print(df.head(5))

input("\n Press Enter to continue.\n")



#Data Cleaning
#Label encode the dataset
df= util.labelEncoder(df, ["HeartDisease","GenHealth","Smoking", "AlcoholDrinking","Sex","PhysicalActivity","AgeCategory"])
 
print("\nHere is a preview of the dataset after label encoding. \n")

print(df.head(5))

input("\nPress Enter to continue.\n")

#One hot encode the dataset
df = util.oneHotEncoder(df, ["Race"])

print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head(5))
                        

input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split

X= df.drop(["HeartDisease"], axis=1)
#axis=1 means that we are dropping the column "HeartDisease" instead of the row
y= df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

from sklearn.tree import DecisionTreeClassifier



clf= DecisionTreeClassifier(max_depth=7, class_weight="balanced")
clf.fit(X_train, y_train)


#6 0.730-0.708=0.022
#7 0.741-0.710=0.031 (I will use this max depth)
#8 0.7185-0.665>0.05 so I will no use this (because is 0.0535)






#Test the model with the testing data set and prints accuracy score

test_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score

test_acc= accuracy_score(y_test, test_predictions)

print("The accuracy with the testing dat set of the Decision Tree Model is: "+ str(test_acc))

#Prints the confusion matrix

from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test, test_predictions, labels=[0,1])

print("\n-----\n")

print("The confusion matrix of the tree is: ")
print(cm)


#Test the model with the training data set and prints accuracy score
train_predictions = clf.predict(X_train)

from sklearn.metrics import accuracy_score

train_acc= accuracy_score(y_train, train_predictions)

print("\n-----\n")

print("The accuracy with the training dat set of the Decision Tree Model is: "+ str(train_acc))
print("\n-----\n")
print("The leaf node with the greatest number of correctly classified patients with Heart Disease are those whose AgeCategory is greater than 5.50. Among these individuals, their general health is rated as 2.50 or lower and further refined to be 1.50 or lower. Additionally, within this subset, patients whose AgeCategory is further refined and found to be greater than 9.50 are considered. Moreover, their Body Mass Index (BMI) falls within the range greater than 16.22 but less than or equal to 40.37. Finally, these patients are not identified as Black, as indicated by their Race_Black value being less than or equal to 0.50. This particular leaf node is the one with the most patients correctly classified with Heart Disease because it has the highest number of weights for the Heart Disease class. Specifically, at this node, the weights are [82.98, 617.82], where 617.82 represents the number of patients correctly classified as having Heart Disease.")

input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations
print("Another application of Decision Trees and considerations:")
print("\n-----\n")
print("When I started doing my makeup, I had a difficult time choosing the right products. Now I love doing my makeup, and I think it is a good idea to implement a Decision Tree in the process of selecting new makeup products. A Decision Tree can enhance makeup recommendations by suggesting products based on a person’s skin type, preferences, and past purchases. For instance, if a user prefers cruelty-free products and has a dry skin type, the tree can recommend products that meet these criteria. To ensure fairness, it’s important to use diverse and representative data to avoid bias, consider individual skin differences, and keep the recommendations updated based on new trends and user feedback.")




#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

util.printTree(clf, X.columns)
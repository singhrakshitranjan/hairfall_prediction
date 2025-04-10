# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load your dataset
# X = ...  # Feature matrix
# y = ...  # Target labels (imbalanced)
dep_data = pd.read_csv('hairfall_data.csv')
X = dep_data.drop('risk',axis=1)
y = dep_data['risk']

# function to add value labels on the class distribution plot
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

# Check class distribution
print("Original class distribution:", Counter(y))

count_class = y.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
addlabels(count_class.index, count_class.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Orginal Class Distribution')
plt.xticks(count_class.index, ['Mild', 'No', 'Moderate', 'High'])
plt.show() # Plot class distribution of original data

# Oversampling using SMOTE 
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X, y)
print("After SMOTE (oversampling) class distribution :", Counter(y_smote))

count_class = y_smote.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
addlabels(count_class.index, count_class.values)
plt.title('Class Distribution after SMOTE')
plt.show() # Plot class distribution after SMOTE

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42) 

# Print class distribution of training set
print("Train set class distribution:", Counter(y_train))

count_class = y_train.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
addlabels(count_class.index, count_class.values)
plt.title('Class Distribution of Training set')
plt.show() # Plot class distribution of the training set

# Define the model and get Cross-Validation score
svm_model = SVC()
scores = cross_val_score(svm_model, X_train, y_train, cv=10)
mean_score = scores.mean()
print(f"Cross Validation Accuracy: {mean_score * 100:.2f}%")

# Train the model
svm_model.fit(X_train,y_train)

# Make predictions
y_pred=svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set Accuracy: {accuracy * 100:.2f}%")


# save the classification model
with open('svm_model_pkl', 'wb') as files:
    pickle.dump(svm_model, files)
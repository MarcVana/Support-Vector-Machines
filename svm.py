"""
Created on Sat Oct  3 18:17:20 2020

SUPPORT VECTOR MACHINE (SVM) PROJECT

Based on Iris dataset.

@author: Marc
"""
# Importing the libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
data = sns.load_dataset('iris')

# Visualizing the species with a pairplot
sns.pairplot(data, hue = 'species')
plt.savefig('species_comparision.png')

# Train and Test splitting
from sklearn.model_selection import train_test_split
X = data.drop('species', axis = 1)
Y = data['species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

# First SVM model
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

# Performing a Grid Search for a better SVM model
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'degree': [1, 2, 3, 4, 5], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid_search = GridSearchCV(SVC(), param_grid, verbose = 3)
grid_search.fit(X_train, Y_train)

# Final SVM model
final_model = grid_search.best_estimator_
final_pred = final_model.predict(X_test)

# Printing the Confusion Matrix and Classification Report for both models
from sklearn.metrics import classification_report, confusion_matrix
print('--------------------------------------------------------')
print('--------------------------------------------------------')
print('------------------FIRST MODEL---------------------------')
print('--------------------------------------------------------')
print('--------------------------------------------------------')
print('Confusion Matrix')
print(confusion_matrix(Y_test, predictions))
print('--------------------------------------------------------')
print('Classification Report')
print(classification_report(Y_test, predictions))
print('--------------------------------------------------------')
print('--------------------------------------------------------')
print('------------------FINAL MODEL---------------------------')
print('--------------------------------------------------------')
print('--------------------------------------------------------')
print('Confusion Matrix')
print(confusion_matrix(Y_test, final_pred))
print('--------------------------------------------------------')
print('Classification Report')
print(classification_report(Y_test, final_pred))
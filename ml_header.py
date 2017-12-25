import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib # image display options

# Linear Regression
from sklearn.linear_model import LinearRegression
# Stochastic Gradient Descent
from sklearn.linear_model import SGDRegressor
# Ridge regression: regularized Linear Regression
from sklearn.linear_model import Ridge
# Least Absolute Shrinkage and Selection Operator Regression: regularized
from sklearn.linear_model import Lasso
# Elastic Net: regularized Linear Regression
from sklearn.linear_model import ElasticNet
# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Stochastic Gradient Descent Classifier
from sklearn.linear_model import SGDClassifier
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
# KNN
from sklearn.neighbors import KNeighborsClassifier

# Support Vector Machine
from sklearn.svm import LinearSVC

# Add polynomial features
from sklearn.preprocessing import PolynomialFeatures

# Cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Confusion Matrix
from sklearn.metrics import confusion_matrix
# Percision and Recall
from sklearn.metrics import precision_score, recall_score
# F1 score
from sklearn.metrics import f1_score
# Percision recall curve
from sklearn.metrics import precision_recall_curve

# Plot learning curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Dataset
from sklearn import datasets
# Download datasets
from sklearn.datasets import fetch_mldata

# Feature scaling
from sklearn.preprocessing import StandardScaler
# Split data in stratified fashion
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
# Copy model
from sklearn.base import clone
# Automation
from sklearn.pipeline import Pipeline

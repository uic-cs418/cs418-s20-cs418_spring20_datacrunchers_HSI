import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def GridCV(X,y):
    #Split data into training and test, skip if you've already done this.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=1)
    #IMPORTANT: Scale your input features so they range from 0 to 1 for a more accurate model. Skip if you've done it.
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)
    #Build model. Skip if done.
    clf = RandomForestRegressor()
    #Add your hyperparameters to test. Whole list of hyperparameters are [n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, ccp_alpha, max_samples]
    para_grids = {
                "n_estimators" : [10,50,100],
                "max_features" : ["auto", "log2", "sqrt"],
                "min_samples_leaf" : [5,10,15,20],
                "bootstrap"    : [True, False]
            }
    
    #Modify cv parameter to change number of folds in your cross validation.
    grid = GridSearchCV(clf, para_grids, scoring = 'mean_squared_error', cv=10)
    grid.fit(X_train, y_train)
    #This is the best fit model for the given hyperparameter trials
    forest_model = grid.best_estimator_
    #Calculate the mean of accuracies across 10 folds of CV.
    mean_accuracy = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10).mean()
    
    return (grid.best_score_, grid.best_params_, mean_accuracy, forest_model)

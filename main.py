import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
from skopt import BayesSearchCV
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
from sklearn.svm import SVC





wine = pd.read_csv('winequality-white.csv', sep=';', header = 'infer')
y = wine.iloc[:,11]
x = wine.drop(wine.columns[11], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=99)

accuracies = []
algorithms = []
times = []
bestparamlist = []

#this part is to get the features so we can modify them with the preprocessors
numeric_features = x.select_dtypes(include=["int64", "float64"]).columns
categorical_features = y

#list of preprocessing methods
preprocessors = {
    ("scaler", StandardScaler()),
    ("MMScaler", MinMaxScaler())

}

#code here inspired from
#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
classifiers = {
            "Random Forest": (RandomForestClassifier(random_state = 99),
            {
            "classifier__bootstrap": [True, False],
            "classifier__max_depth": [25, 50, None], #default is none
            "classifier__n_estimators": [25, 50, 75, 100, 125, 150], #higher seemingly always better, default is 100
            "classifier__min_samples_split": [2, 4, 6, 8], #default values (the first one) seems to always be chosen
            "classifier__min_samples_leaf": [1, 3, 5, 7] #default values (the first one) seems to always be chosen
            }),
            "K-Nearest-Neighbors": (KNeighborsClassifier(),
            {
            "classifier__n_neighbors": [3, 4, 5, 6, 10], #default 5 used small numbers for this because it never picks the big ones
            "classifier__leaf_size": [10, 30, 50, 100, 200], #in this problem leaf size doesnt seem to affect accuracy
            "classifier__algorithm": ['ball_tree', 'kd_tree', 'brute', 'auto'],
            "classifier__p": [1, 2],
            "classifier__weights": ['uniform', 'distance']
            }),
            "Support Vector Machine": (SVC(random_state = 99),
            {
            "classifier__C": [1, 10, 100],
            "classifier__kernel": ['rbf', 'sigmoid'] #other kernals just run endlessly, or atleast longer than 15~ minutes
            })

    }


################################################################################################################################################################################
#Optimized
################################################################################################################################################################################
for clf, (classifiers, params) in classifiers.items():
        #pipeline = Pipeline([("classifier", classifiers)])
        #for each preprocessing method
        for tf in preprocessors:
            #get the name of the preprocessing method
            ppName = str(tf)
            ppName = (re.search("\'.*\'", ppName)).group()
            #this bit is from
            #https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
            numeric_transformer = Pipeline(steps=[
                tf
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features)
                    #('cat', categorical_transformer, categorical_features)
            ])
            pipeline = Pipeline([("preprocessor", preprocessor),("classifier", classifiers)])
################################################################################################################################################################################
#Default Hyperparameters
################################################################################################################################################################################
            t0 = time.time()
            pipeline.fit(x_train, y_train)
            t1 = time.time()
            Acc = pipeline.score(x_test, y_test)
            times.append(t1-t0)
            accuracies.append(Acc * 100)
            algorithms.append(clf + " Default " + "(" + ppName + ")")
################################################################################################################################################################################
#Hyperparameter Optimization
################################################################################################################################################################################
            bayesSearch = BayesSearchCV(pipeline, params, n_jobs = 15, scoring = 'accuracy', verbose = 0, random_state = 99, n_iter = 15, cv = 5, return_train_score = True)
            t0 = time.time()
            bayesSearch.fit(x_train, y_train)
            t1 = time.time()
            best_params = bayesSearch.best_params_
            bestparamlist.append("Best Paremeters found for {} ({}): {}".format(clf, ppName, best_params))
            Acc = bayesSearch.score(x_test, y_test)
            times.append(t1-t0)
            accuracies.append(Acc * 100)
            algorithms.append(clf + " HPO " + "(" + ppName + ")")
################################################################################################################################################################################



#had to be remade for some reason
classifiers = {
            "Random Forest": (RandomForestClassifier(random_state = 99),
            {
            "classifier__bootstrap": [True, False],
            "classifier__max_depth": [25, 50, None], #default is none
            "classifier__n_estimators": [25, 50, 75, 100, 125, 150], #higher seemingly always better, default is 100
            "classifier__min_samples_split": [2, 4, 6, 8], #default values (the first one) seems to always be chosen
            "classifier__min_samples_leaf": [1, 3, 5, 7] #default values (the first one) seems to always be chosen
            }),
            "K-Nearest-Neighbors": (KNeighborsClassifier(),
            {
            "classifier__n_neighbors": [3, 4, 5, 6, 10], #default 5 used small numbers for this because it never picks the big ones
            "classifier__leaf_size": [10, 30, 50, 100, 200], #in this problem leaf size doesnt seem to affect accuracy
            "classifier__algorithm": ['ball_tree', 'kd_tree', 'brute', 'auto'],
            "classifier__p": [1, 2],
            "classifier__weights": ['uniform', 'distance']
            }),
            "Support Vector Machine": (SVC(random_state = 99),
            {
            "classifier__C": [1, 10, 100],
            "classifier__kernel": ['rbf', 'sigmoid'] #other kernals just run endlessly, or atleast longer than 15~ minutes
            })

    }

################################################################################################################################################################################
#unoptimized
################################################################################################################################################################################
for clf, (classifiers, params) in classifiers.items():
    pipeline = Pipeline([("classifier", classifiers)])
################################################################################################################################################################################
#Default Hyperparameters
################################################################################################################################################################################
    t0 = time.time()
    pipeline.fit(x_train, y_train)
    t1 = time.time()
    Acc = pipeline.score(x_test, y_test)
    times.append(t1-t0)
    accuracies.append(Acc * 100)
    algorithms.append(clf + " Default (No Preprocessing)")
################################################################################################################################################################################
#Hyperparameter Optimization
################################################################################################################################################################################
    bayesSearch = BayesSearchCV(pipeline, params,n_jobs = 15, scoring = 'accuracy', verbose = 0, random_state = 99, n_iter = 15, cv = 5, return_train_score = True)
    t0 = time.time()
    bayesSearch.fit(x_train, y_train)
    t1 = time.time()
    best_params = bayesSearch.best_params_
    bestparamlist.append("Best Paremeters found for {} (No Preprocessing): {}".format(clf, best_params))
    Acc = bayesSearch.score(x_test, y_test)
    times.append(t1-t0)
    accuracies.append(Acc * 100)
    algorithms.append(clf + " HPO (No Preprocessing)")
################################################################################################################################################################################
    
################################################################################################################################################################################


plt.figure(figsize = (10,10))
plt.barh(algorithms, accuracies, align = 'edge', height = 0.4)
plt.barh(algorithms, times, align = 'edge', height = -0.4)

plt.legend({'Accuracy (Percent)':'blue', 'Time (Seconds)':'orange'}, loc='lower right', bbox_to_anchor=(1.25, 0))
plt.xlabel("Accuracy and Time")
plt.ylabel("Algorithms")
plt.title("Algorithms with Accuracy/Time")
plt.show()

#code for this from https://stackoverflow.com/questions/48053979/print-2-lists-side-by-side user SCB
sortedAlgList = "\n".join("{}: {:0.5f}% accuracy, {:0.5f} seconds".format(y, x, z) for x, y, z in sorted(zip(accuracies, algorithms, times), key = lambda x: (x[0], -x[2]), reverse = True))
print("Best Algorithm by accuracy and time is:", sortedAlgList.partition(":")[0], "\n")
print("List of algorithms sorted best to worst:")
print(sortedAlgList)
print("\nList of best hyperparameters per algorithm:")
bestparamlist = "\n".join(bestparamlist)
print(bestparamlist)

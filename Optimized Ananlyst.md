                                        Optimized 2 classification models predicting Top 50 Flag
                                        
A. Random Forest with RandomizedSearchCV
1. Setting
         "n_estimators": randint(100, 500),
        "max_depth": [None, 8, 10, 12, 15, 20],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 6),
        "max_features": ["sqrt", "log2"],
2. Result
a. Before tuning

   Train accuracy: 1.0000
   Test accuracy: 0.9934
b. After tunning

   Train accuracy: 0.9983
   Test accuracy: 0.9900
   
Why RandomizedSearchCV
+ RF has large tuning space
+ RF explores broadly with less computation than exhaustive

A. Logistic Regression with GridSearchCV
1. Setting
         param_grid={
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"],
    },
2. Result
a. Before tuning

   Train accuracy: 0.9875
   Test accuracy: 0.9834
b. After tunning

   Train accuracy: 0.9942
   Test accuracy: 0.9934
   
Why RandomizedSearchCV
+ The parameter space is small
+ Grid search checks every combination, so it is exhaustive and reliable for this model
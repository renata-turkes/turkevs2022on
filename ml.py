# from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import model





def tune_hyperparameters(data_train, labels_train):      
    
    # Classification or regression problem.
    if "int" in str(type(labels_train[0])) or "str" in str(type(labels_train[0])):   
        classifier = SVC()    
    else:  
        classifier = SVR()      

    
    pipeline = Pipeline([# ("Scaler", preprocessing.StandardScaler()),
                         ("classifier", classifier)]) 
    
    param_grid = [{"classifier":     [classifier],
                   "classifier__C":  [0.001, 1, 100]}]   
     
    best_ml_model, _ = model.grid_search(data_train, labels_train, param_grid, pipeline)
    return best_ml_model   



# def tune_hyperparameters(data_train, labels_train):      
    
#     if "int" in str(type(labels_train[0])) or "str" in str(type(labels_train[0])):   
#         param_grid = [{"classifier":    [SVC()],
#                       "classifier__C":  [0.001, 1, 100]},                     
                                    
#                       {"classifier":    [LogisticRegression()], # LogisticRegression(max_iter = 100000)
#                        "classifier__C": [0.001, 1, 100]},
                  
#                       {"classifier":    [KNeighborsClassifier()]}]     
#     else:      
#         param_grid = [{"classifier":    [SVR()],
#                       "classifier__C":  [0.001, 1, 100]},                     
                                    
#                       {"classifier":    [LinearRegression()]}]        
    
#     pipeline = Pipeline([# ("Scaler", preprocessing.StandardScaler()),
#                          ("classifier", SVC())])   
     
#     best_ml_model, _ = model.grid_search(data_train, labels_train, param_grid, pipeline)    
#     return best_ml_model   
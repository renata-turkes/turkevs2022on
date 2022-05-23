import numpy as np
import gudhi as gd
import gudhi.representations
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

import ph
import data_construction
import model



# Practical remark about GUDHI persistence signatures.
# PL sample_range is the minimum and maximum of all piecewise-linear function domains (the interval on which samples will be drawn evenly).
# If sample range is not given, it is calculated within .fit() as the minimum and maximum value of PDs.
# However, we fit PLs separately across transformations, so that if e.g. PD=[0, 10] changes under noise to [0, 20], 
# the base of the two PLs could span across the whole range on the x-axis.
# To properly reflect the change in PDs in their PLs, the range to plot the bases should be the same across all transformations!
# PL = gd.representations.Landscape(sample_range = [min_birth, max_death])    
# Similarly as with PLs above, the im_range needs to be explicitly given when fitting PIs, in order for the change in PDs under transformations
# to be properly reflected in the PIs.
# PI = gd.representations.PersistenceImage(im_range = [min_birth, max_birth, min_lifespan, max_lifespan]

def tune_hyperparameters(pds_train, labels_train, min_b = 0, max_b = 2, min_d = 0, max_d = 2, min_l = 0, max_l = 2):      
    
    # Classification problem.
    if "int" in str(type(labels_train[0])) or "str" in str(type(labels_train[0])):   
        classifier = SVC()        
        
    # Regression problem.
    else:
        classifier = SVR()
    
    pipeline = Pipeline([("PH", gd.representations.PersistenceImage(resolution = [10, 10], im_range = [min_b, max_b, min_l, max_l])),
                         # ("Scaler", preprocessing.StandardScaler()),
                         ("classifier", classifier )])  
    
    pds_nums_cycles = [len(pd) for pd in pds_train]
    max_num_cycles = max(pds_nums_cycles)
    
    param_grid = [{"PH":                  [FunctionTransformer()],
                   "PH__func":            [ph.sorted_lifespans_pds],
                   "classifier":          [classifier],
                   "classifier__C":       [0.001, 1, 100]},              

                  {"PH":                  [gd.representations.PersistenceImage(resolution = [10, 10], im_range = [min_b, max_b, min_l, max_l])],
                   "PH__bandwidth":       [0.1, 0.5, 1, 10],
                   "PH__weight":          [lambda x: 1, lambda x: x[1], lambda x: x[1]**2],
                   "classifier":          [classifier],
                   "classifier__C":       [0.001, 1, 100]},
                  
                  {"PH":                   [gd.representations.Landscape(resolution = 100, sample_range = [min_b, max_d])],
                   "PH__num_landscapes":   [1, 10, max_num_cycles],
                   "classifier":           [classifier], 
                   "classifier__C":        [0.001, 1, 100]}]
     
    best_ph_ml_pipeline, grid_search = model.grid_search(pds_train, labels_train, param_grid, pipeline)    
    
    # This returns an error if the best signature is persistence image, since it's parameter is a function lambda:
    # AttributeError: Can't pickle local object 'tune_hyperparameters.<locals>.<lambda>'.
    # with open(PATH_CURRENT + "results/best_ph_ml_pipeline.pkl", "wb") as f:
        # pickle.dump(best_ph_ml_pipeline, f)
        
    # Here we also store the best PH (and not only PH+ML) pipeline, in order to be able to retrieve and visualize PH information.
    # best_ph_pipeline = clone(grid_search.best_params_["PH"]) 
    # with open(PATH_CURRENT + "resuls/best_ph_pipeline.pkl", "wb") as f:
    #     pickle.dump(best_ph_pipeline, f)  
    
    return best_ph_ml_pipeline   
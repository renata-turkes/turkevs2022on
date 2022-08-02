import numpy as np
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
# import keras_tuner as kt

import model





def build_model_classification(num_features, num_classes, depth, input_layer_width, hidden_layers_width, learning_rate):
    model = keras.models.Sequential()
    
    # Input layer.
    model.add(keras.layers.Dense(input_layer_width, input_dim = num_features, activation = "relu")) 
    
    # Hidden layers.
    for l in range(depth):
        model.add(keras.layers.Dense(hidden_layers_width, activation = "relu"))        
    
    # Output layer.
    model.add(keras.layers.Dense(num_classes, activation = "softmax"))    

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
                  loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False),
                  metrics = ["accuracy"])    
    return model



def build_model_regression(num_features, depth, input_layer_width, hidden_layers_width, learning_rate):
    model = keras.models.Sequential()
    
    # Input layer.
    model.add(keras.layers.Dense(input_layer_width, input_dim = num_features, activation = "relu")) 
    
    # Hidden layers.
    for l in range(depth):
        model.add(keras.layers.Dense(hidden_layers_width, activation = "relu"))        
    
    # Output layer.
    model.add(keras.layers.Dense(1))        
    
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
                  loss = "mean_squared_error", 
                  metrics = ["mean_squared_error"])  
    return model



def tune_hyperparameters(data_train, labels_train, min_depth = 1, max_depth = 5): 
    
    num_features = data_train.shape[1]
    
    # Classification problem.
    if "int" in str(type(labels_train[0])) or "str" in str(type(labels_train[0])):
        problem = "classification"
        num_classes = len(np.unique(labels_train))
        nn_deep = KerasClassifier(build_fn = build_model_classification, 
                                  epochs = 2, 
                                  verbose = 0,
                                  num_features = num_features, 
                                  num_classes = num_classes)   
    # Regression problem.
    else:
        problem = "regression"
        nn_deep = KerasRegressor(build_fn = build_model_regression, 
                                 epochs = 2, 
                                 verbose = 0,
                                 num_features = num_features)

    depths = np.linspace(min_depth, max_depth, num = 1, dtype = int)
    log = 12 # int(np.log2(num_features))   
    # widths = np.power(2, [log-6, log-4, log-2, log]) 
    widths = np.power(2, [log-6, log-4, log-2, log])        
    learning_rates = [0.01, 0.001]  
    print("NN deep hyperparameters:")
    print("depths = ", depths)
    print("layer_width = ", widths)
    print("learning_rates = ", learning_rates)    
    param_grid = {"depth": depths,
                  "input_layer_width": widths,
                  "hidden_layers_width": widths,
                  "learning_rate": learning_rates}    
    best_nn_deep, grid_search = model.grid_search(data_train, labels_train, param_grid, nn_deep) 
    
    # Do not return KerasClassifier instance, since fit() will then always build a NEW MODEL to train.
    # We need to return Keras model instance with the best parameters.)        
    depth_best = grid_search.best_params_["depth"]
    input_layer_width_best = grid_search.best_params_["input_layer_width"]
    hidden_layers_width_best = grid_search.best_params_["hidden_layers_width"]
    learning_rate_best = grid_search.best_params_["learning_rate"]

    if problem == "classification":
        best_nn_deep = build_model_classification(num_features, num_classes, depth_best, input_layer_width_best, hidden_layers_width_best, learning_rate_best)
    else:
        best_nn_deep = build_model_regression(num_features, depth_best, input_layer_width_best, hidden_layers_width_best, learning_rate_best)
    
    return best_nn_deep  
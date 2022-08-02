import numpy as np
import pandas as pd
import collections
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.base import clone 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import time
import sys

import plots





def fit(data_train, labels_train, model, num_train_iter = -1):    
    # <class 'sklearn.pipeline.Pipeline'> 
    # <class 'keras.wrappers.scikit_learn.KerasClassifier'>
    # <class 'keras.engine.functional.Functional'>
    if "keras" in str(type(model)):           
        history = model.fit(data_train, labels_train, epochs = abs(num_train_iter), validation_split = 0.2, verbose = 0) 
    if "sklearn" in str(type(model)):
        model = model.fit(data_train, labels_train) 
        history = None        
    return model, history 
    


# For classification problems, score = classification accuracy, and greater is better.
# For regression, score = mean squared error, and lower is better.
def get_score(data_test, labels_test, model):  \

    if "keras" in str(type(model)):    
        _, score_test = model.evaluate(data_test, labels_test, verbose = 0)    
        
    if "sklearn" in str(type(model)):       
        
        # Classification problem.
        if "int" in str(type(labels_test[0])) or "str" in str(type(labels_test[0])):  
            score_test = model.score(data_test, labels_test)  
            
        # Regression problem.
        else: 
            predictions = model.predict(data_test)
            score_test = mean_squared_error(labels_test, predictions)   
            
    return score_test



def get_scores_under_trnsfs(data_trnsfs, trnsfs, labels, model):
    accs = []
    for trnsf in trnsfs:
        data = data_trnsfs[trnsf]             
        acc = get_score(data, labels, model)
        accs.append(acc)    
    return accs

  
    
def clone(model, problem = "classification"):   
    if "sklearn" in str(type(model)) or "scikit_learn" in str(type(model)):
        model_cloned = sklearn.base.clone(model) 
    else:
        model_cloned = tf.keras.models.clone_model(model)
        
        if problem == "classification": 
            model_cloned.compile(optimizer = keras.optimizers.Adam(learning_rate = keras.backend.eval(model.optimizer.lr)),
                                  loss = "sparse_categorical_crossentropy",
                                  metrics = ["accuracy"])  
            
        elif problem == "regression":
            model_cloned.compile(optimizer = keras.optimizers.Adam(learning_rate = keras.backend.eval(model.optimizer.lr)),
                                  loss = "mean_squared_error",
                                  metrics = ["mean_squared_error"])  
    
    return model_cloned
    
    
    
def grid_search(data, labels, param_grid, model): 
    print("Tuning the hyperparameters with GridSearchCV() ...")
    # print("data.shape = ", data.shape)
    # print("labels.shape = ", labels.shape)    
    # print("Numbers of point clouds with each label value: ", collections.Counter(labels))
    t0 = time.time()  
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1)
    grid_search = grid_search.fit(data, labels)
    t1 = time.time()
    print("Runtime = ", np.around(t1-t0, 2), " seconds.")

    accuracies_data_frame =  pd.DataFrame({"rank": grid_search.cv_results_["rank_test_score"], 
                                           "param": grid_search.cv_results_["params"], 
                                           "avg acc": grid_search.cv_results_["mean_test_score"]}).sort_values(by = "rank").head(10)
    print(accuracies_data_frame) # display(accuracies_data_frame)         
    print("Best parameters:", grid_search.best_params_)
    print("Best accuracy = ", grid_search.best_score_)   
    
    # Return unfitted model.
    # Classification problem.
    if "int" in str(type(labels[0])) or "str" in str(type(labels[0])):  
        best_model = clone(grid_search.best_estimator_, problem = "classification")
    else:
        best_model = clone(grid_search.best_estimator_, problem = "regression")    
    return best_model, grid_search



def fit_and_plot_learning_curve(data_train, labels_train, model, num_train_iter = 3, train_sizes = np.linspace(0.1, 1.0, 5), num_splits = 3, output_path = " "):
    
    # Classification or regression problem.
    if "int" in str(type(labels_train[0])) or "str" in str(type(labels_train[0])):  
        problem = "classification"
    else:
        problem = "regression"       

    data_train, data_val, labels_train, labels_val = sklearn.model_selection.train_test_split(data_train, labels_train, test_size = 0.2, random_state = 42)
         
    num_train_sizes = len(train_sizes)
    scores_train = np.zeros((num_train_sizes, num_splits))
    scores_test = np.zeros((num_train_sizes, num_splits)) 
    num_samples = len(data_train)
    for t, train_size in enumerate(train_sizes):
        
        train_size_abs = (train_size * num_samples).astype(int) 
        scores_split_train = []
        scores_split_test = []
        for split in range(num_splits):  
        
            indices_train = np.random.choice(np.arange(num_samples), size = train_size_abs) 
            data_split_train = data_train[indices_train] 
            labels_split_train = labels_train[indices_train]        
        
            # print("indices_train = ", indices_train)
            # print("data_split_train.shape = ", data_split_train.shape)
            # print("labels_split_train.shape = ", labels_split_train.shape)
            # print("Numbers of point clouds with each label value in labels_train: ", collections.Counter(labels_split_train))
            # print("data_val.shape = ", data_val.shape)
            # print("labels_val.shape = ", labels_val.shape)
            # print("Numbers of point clouds with each label value in labels_val: ", collections.Counter(labels_val))        
            # print("deafult batch size = 32, so that num_batches = ", np.ceil(data_split_train.shape[0] / 32))
            
            model_trained, _ = fit(data_split_train, labels_split_train, clone(model, problem), num_train_iter)
            
            acc_split_train = get_score(data_split_train, labels_split_train, model_trained)
            acc_split_val = get_score(data_val, labels_val, model_trained)         
            # print("acc_split_train = ", np.around(acc_split_train, 2))
            # print("acc_split_val = ", np.around(acc_split_val, 2))            
            
            scores_train[t][split] = acc_split_train
            scores_test[t][split] = acc_split_val       
    
    scores_mean_train = np.mean(scores_train, axis = 1)
    scores_std_train = np.std(scores_train, axis = 1)
    scores_mean_test = np.mean(scores_test, axis = 1)
    scores_std_test = np.std(scores_test, axis = 1)    
    num_samples = len(data_train)
    nums_train_samples = train_sizes * num_samples   
    plots.plot_curves(x_vals = nums_train_samples, y1_vals = scores_mean_train, y2_vals = scores_mean_test,
                      y1_std = scores_std_train, y2_std = scores_std_test, y1_label = "train", y2_label = "validation", 
                      xlabel = "number of training point clouds", ylabel = "accuracy", output_path = output_path) # title = "Learning curve"
    
    
    
def fit_and_plot_learning_curve_sklearn(data_train, labels_train, model, train_sizes = np.linspace(0.1, 1.0, 5), num_splits = 3, output_path = " "):    
    
    # Classification or regression problem.
    if "int" in str(type(labels_train[0])) or "str" in str(type(labels_train[0])):  
        problem = "classification"
    else:
        problem = "regression"       
    model = clone(model, problem) 
    
    # Here, num_epochs = 1?
    nums_train_samples, scores_train, scores_test = sklearn.model_selection.learning_curve(estimator = model, 
                                                                                           X = data_train, 
                                                                                           y = labels_train, 
                                                                                           cv = num_splits,    
                                                                                           train_sizes = np.linspace(0.1, 1.0, 3), 
                                                                                           n_jobs = -1)     
    scores_mean_train = np.mean(scores_train, axis = 1)
    scores_std_train = np.std(scores_train, axis = 1)
    scores_mean_test = np.mean(scores_test, axis = 1)
    scores_std_test = np.std(scores_test, axis = 1)       
    plots.plot_curves(x_vals = nums_train_samples, y1_vals = scores_mean_train, y2_vals = scores_mean_test,
                      y1_std = scores_std_train, y2_std = scores_std_test, y1_label = "train", y2_label = "validation", 
                      xlabel = "number of training point clouds", ylabel = "accuracy", title = "Learning curve (sklearn)")
    
    

def fit_and_plot_training_curve(data_train, labels_train, model, num_epochs = 25, batch_size = 32, output_path = " "): 
    
    # Classification or regression problem.
    if "int" in str(type(labels_train[0])) or "str" in str(type(labels_train[0])):  
        problem = "classification"
    else:
        problem = "regression"       
    model = clone(model, problem) 
    
    data_train, data_val, labels_train, labels_val = sklearn.model_selection.train_test_split(data_train, labels_train, test_size = 0.2, random_state = 42)
    
    num_samples = len(data_train)
    num_batches = int(num_samples / batch_size)
    # print("num_epochs = ", num_epochs)
    # print("batch_size = ", batch_size)
    # print("num_samples = ", num_samples)
    # print("num_batches = ", num_batches)
    scores_train = np.zeros((num_epochs, num_batches))
    scores_test = np.zeros((num_epochs, num_batches))    
    
    for epoch in range(num_epochs):
        scores_batch_train = []
        scores_batch_test = []
        random_perm = np.random.permutation(num_samples)
        sample_start_index = 0
        for batch in range(num_batches):
            # print("\nepoch = ", epoch+1, ", batch = ", batch+1)
            
            indices = random_perm[sample_start_index : sample_start_index + batch_size]
            data_batch_train = data_train[indices]
            labels_batch_train = labels_train[indices]

            model, _ = fit(data_batch_train, labels_batch_train, model, num_train_iter = -1) 
             
            acc_batch_train = get_score(data_batch_train, labels_batch_train, model)
            acc_batch_test = get_score(data_val, labels_val, model)
            # print("acc_batch_train = ", np.around(acc_batch_train, 2))
            # print("acc_batch_test = ", np.around(acc_batch_test, 2))
            
            scores_batch_train.append(acc_batch_train)
            scores_batch_test.append(acc_batch_test)            
            sample_start_index = sample_start_index + batch_size
        scores_train[epoch] = np.asarray(scores_batch_train)
        scores_test[epoch] = np.asarray(scores_batch_test)        
    
    scores_mean_train = np.mean(scores_train, axis = 1)
    scores_std_train = np.std(scores_train, axis = 1)
    scores_mean_test = np.mean(scores_test, axis = 1)
    scores_std_test = np.std(scores_test, axis = 1)       
    nums_epochs = np.arange(num_epochs)
    plots.plot_curves(x_vals = nums_epochs, y1_vals = scores_mean_train, y2_vals = scores_mean_test, 
                      y1_std = scores_std_train, y2_std = scores_std_test, y1_label = "train", y2_label = "validation", 
                      xlabel = "number of epochs", ylabel = "accuracy", output_path = output_path) # title = "Training curve"
    
    # return scores_mean_train, scores_std_train, scores_mean_train, scores_std_test    
    return model

    
    
def plot_training_curve_keras(history, num_epochs, output_path):
    plots.plot_curves(x_vals = np.arange(num_epochs), 
                      y1_vals = np.asarray(history.history["accuracy"]), 
                      y2_vals = np.asarray(history.history["val_accuracy"]), 
                      y1_std = 0, 
                      y2_std = 0, 
                      y1_label = "train", 
                      y2_label = "validation", 
                      xlabel = "epoch", 
                      ylabel = "accuracy", 
                      title = "Training curve (keras)", 
                      output_path = output_path)        
        
    

def plot_confusion_matrix(data, labels, label_encoder, model, output_path = " "):
    predictions = model.predict(data)    
    
    if "sklearn" in str(type(model)):   
        
        if len(np.unique(labels)) == 2:
            # sklearn model on a binary classification problem returns a non-integer value, for each sample.   
            predictions = (predictions > 0.5).astype(int)
    
    
    elif "keras" in str(type(model)):          
        
        if len(np.unique(labels)) == 2:
            # keras model on a binary classification problem returns a vector of non-integer value, for each sample.
            predictions = np.asarray([predictions[i][0] for i in range(len(predictions))]) # e.g. [[0.1], [0.5], [-0.2]] -> [0.1, 0.5, -0.2]
            predictions = (predictions < 0.5).astype(int)
            
        else:
            # keras model on a multi-class classification problem returns a vector of probabilities, for each sample.
            predictions = np.argmax(predictions, axis = 1) 
    
    if label_encoder != None:
        predictions = label_encoder.inverse_transform(predictions)
    
    # print("\nConfusion matrix:")
    conf_mat = sklearn.metrics.confusion_matrix(labels, predictions) # conf_matrix = tf.math.confusion_matrix(labels_test, predictions).numpy()
      
    display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = np.unique(labels)) 
    display = display.plot()
    fig = display.figure_    
    if output_path != " ":
        fig.savefig(output_path, bbox_inches = "tight") 
    plt.show()
    plt.clf()
    # return conf_mat
    
  
    
def visualize_incorrect_predictions(data_pc, data, labels, label_encoder, model, output_path = " "):
    predictions = model.predict(data)

    if "sklearn" in str(type(model)):   
        
        if len(np.unique(labels)) == 2:
            # sklearn model on a binary classification problem returns a non-integer value, for each sample.   
            predictions = (predictions > 0.5).astype(int)
    
    elif "keras" in str(type(model)):          
        
        if len(np.unique(labels)) == 2:
            # keras model on a binary classification problem returns a vector of non-integer value, for each sample.
            predictions = np.asarray([predictions[i][0] for i in range(len(predictions))]) # e.g. [[0.1], [0.5], [-0.2]] -> [0.1, 0.5, -0.2]
            predictions = (predictions < 0.5).astype(int)
            
        else:
            # keras model on a multi-class classification problem returns a vector of probabilities, for each sample.
            predictions = np.argmax(predictions, axis = 1)
    
    if label_encoder != None:
        predictions = label_encoder.inverse_transform(predictions) 
        
    # print("Labels are incorrectly predicted for the following point clouds: \n\n")
    # num_incorrect_predictions = 0
    # for pc, prediction, label in zip(data_pc, predictions, labels):
    #     if prediction != label:
    #         num_incorrect_predictions = num_incorrect_predictions + 1
    #         plots.plot_point_cloud(pc, title = "label=%d, prediction=%d" %(label, prediction)) 
    # num_samples = len(data_pc)
    # perc_incorrect_predictions = num_incorrect_predictions / num_samples
    # print("Percenatage of incorrect predictions = ", np.around(100 * perc_incorrect_predictions, 2), "%.")     

    # print("Labels are incorrectly predicted for the following point clouds:")
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_path + ".pdf")
    num_incorrect_predictions = 0
    for pc, prediction, label in zip(data_pc, predictions, labels):
        if prediction != label:
            num_incorrect_predictions = num_incorrect_predictions + 1
            fig = plots.plot_point_cloud(pc, title = "label=%d, prediction=%d" %(label, prediction)) 
            pdf.savefig(fig)
            plt.clf()
    pdf.close()
    num_samples = len(data_pc)
    perc_incorrect_predictions = num_incorrect_predictions / num_samples
    print("Percenatage of incorrect predictions = ", np.around(100 * perc_incorrect_predictions, 2), "%. \n")      
    
    
    
def plot_regression_line(data, labels, model):
    predictions = model.predict(data) 
    fig, axes = plt.subplots(figsize = (3, 3))  
    axes.scatter(labels, predictions)
    axes.set_xlabel("True label", fontsize = 20)
    axes.set_ylabel("Predicted label", fontsize = 20)
    axes.set_xlim(np.min(labels), np.max(labels))
    axes.set_ylim(np.min(labels), np.max(labels))
    axes.set_aspect("equal")
    return fig, axes



def plot_regression_line_unlimited(data, labels, model):
    predictions = model.predict(data) 
    fig, axes = plt.subplots(figsize = (3, 3))  
    axes.scatter(labels, predictions)
    axes.set_xlabel("True label")
    axes.set_ylabel("Predicted label")
    axes.set_aspect("equal")
    return fig, axes
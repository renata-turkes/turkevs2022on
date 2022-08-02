# The implementation is based on the model available at https://keras.io/examples/vision/pointnet/.



import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error

import model





def build_model_classification(num_points, num_classes, num_filters = 64, learning_rate = 0.001):
   
    inputs = keras.Input(shape=(num_points, 3))
   
    # Input transformer N x n x 3 -> N x n x 3. TNet without regularization.
    x = tnet(inputs, 3)
    
    # Embed to 64-dim space (N x n x 3 -> N x n x 64).
    x = conv_bn(x, num_filters)
    x = conv_bn(x, num_filters)
    
    # Feature transformer (N x n x 64 -> N x n x 64). TNet with regularization.
    x = tnet(x, num_filters)
    
    # Embed to 1024-dim space (N x n x 64 -> N x n x 1024).
    x = conv_bn(x, num_filters)
    x = conv_bn(x, 2 * num_filters)
    x = conv_bn(x, 16 * num_filters)
       
    # Global feature vector (N x n x 1024 -> N x 1024).
    x = keras.layers.GlobalMaxPooling1D()(x)
    
    # Fully connected layers to output k = num_classes scores (N x 1024 -> N x k).
    x = dense_bn(x, 8 * num_filters)
    x = keras.layers.Dropout(0.3)(x)
    x = dense_bn(x, 4 * num_filters)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs = inputs, outputs = outputs, name = "point_net_classification")    
        
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"]) # metrics = ["sparse_categorical_accuracy"]
       
    return model



def build_model_regression(num_points, num_filters = 64, learning_rate = 0.001):
   
    inputs = keras.Input(shape=(num_points, 3))
   
    # Input transformer N x n x 3 -> N x n x 3. TNet without regularization.
    x = tnet(inputs, 3)
    
    # Embed to 64-dim space (N x n x 3 -> N x n x 64).
    x = conv_bn(x, num_filters)
    x = conv_bn(x, num_filters)
    
    # Feature transformer (N x n x 64 -> N x n x 64). TNet with regularization.
    x = tnet(x, num_filters)
    
    # Embed to 1024-dim space (N x n x 64 -> N x n x 1024).
    x = conv_bn(x, num_filters)
    x = conv_bn(x, 2 * num_filters)
    x = conv_bn(x, 16 * num_filters)
    
    # Global feature vector (N x n x 1024 -> N x 1024).
    x = keras.layers.GlobalMaxPooling1D()(x)
    
    # Fully connected layers to output k = num_classes scores (N x 1024 -> N x k).
    x = dense_bn(x, 8 * num_filters)
    x = keras.layers.Dropout(0.3)(x)
    x = dense_bn(x, 4 * num_filters)
    x = keras.layers.Dropout(0.3)(x)    
    outputs = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs = inputs, outputs = outputs, name = "point_net_regression")        
        
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
                  loss = "mean_squared_error", 
                  metrics = ["mean_squared_error"])    
       
    return model



@tf.keras.utils.register_keras_serializable(package='Custom', name='orthogonal') # Added by author so that tf.keras.models.clone_model() in model.py can work.
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
    # Added by author so that tf.keras.models.clone_model() in model.py can work.
    # https://github.com/keras-team/keras/blob/v2.7.0/keras/regularizers.py#L46-L207
    # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer
    def get_config(self):
        return {'num_features': int(self.num_features), 'l2reg': float(self.l2reg)}
    
    

def tnet(inputs, num_features):

    # Embed to higher dimension.
    x = conv_bn(inputs, num_features)
    x = conv_bn(x, 2 * num_features)
    x = conv_bn(x, 16 * num_features)
    
    # Global features.
    x = keras.layers.GlobalMaxPooling1D()(x)
    
    # Fully connected layers.
    x = dense_bn(x, 8 * num_features)
    x = dense_bn(x, 4 * num_features)
    
    # Convert to K x K matrix.
    # Add bias and regularization.
    bias = keras.initializers.Constant(np.eye(num_features).flatten()) # Initalise bias as the indentity matrix.
    reg = OrthogonalRegularizer(num_features)
    x = keras.layers.Dense(num_features * num_features,
                           kernel_initializer = "zeros",
                           bias_initializer = bias,
                           activity_regularizer = reg,)(x)    

    # Apply affine transformation to input features.
    feat_T = keras.layers.Reshape((num_features, num_features))(x)
    return keras.layers.Dot(axes=(2, 1))([inputs, feat_T])

    
    
def conv_bn(x, filters):
    x = keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.Activation("relu")(x)



def dense_bn(x, filters):
    x = keras.layers.Dense(filters)(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.Activation("relu")(x)



def tune_hyperparameters(data_train, labels_train):    
    num_points = data_train.shape[1]
    
    # Classification problem.
    if "int" in str(type(labels_train[0])) or "str" in str(type(labels_train[0])):
        problem_type = "classification"
        num_classes = len(np.unique(labels_train))
        point_net = KerasClassifier(build_fn = build_model_classification, 
                                     epochs = 2, 
                                     verbose = 0,
                                     num_points = num_points, 
                                     num_classes = num_classes)   
    # Regresssion problem.
    else:
        problem_type = "regression"
        point_net = KerasRegressor(build_fn = build_model_regression, 
                                   epochs = 2, 
                                   verbose = 0,
                                   num_points = num_points)
        
    # Parameters.
    nums_filters = np.power(2, [4, 5, 6]) # [2, 4, 5, 6]
    learning_rates = [0.01, 0.001]  
    print("PointNet hyperparameters:")
    print("nums_filters = ", nums_filters)
    print("learning_rates = ", learning_rates)
    param_grid = {"num_filters": nums_filters, 
                  "learning_rate": learning_rates}       
    best_point_net, grid_search = model.grid_search(data_train, labels_train, param_grid, point_net)
        
    # Do not return KerasClassifier instance, since fit() will then always build a NEW MODEL to train.
    # We need to return Keras model instance with the best parameters.        
    num_filters_best = grid_search.best_params_["num_filters"]
    learning_rate_best = grid_search.best_params_["learning_rate"]        
    if problem_type == "classification":
        best_point_net = build_model_classification(num_points, num_classes, num_filters = num_filters_best, learning_rate = learning_rate_best)
    else:
        best_point_net = build_model_regression(num_points, num_filters = num_filters_best, learning_rate = learning_rate_best)
    return best_point_net  
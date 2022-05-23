print("\n\nCONVEXITY: Point clouds binary classification via persistent homology (PH) and/or deep learning (DL). \n\n")
print("train data = random shapes")
print("test data = random shapes")
# Train and test indices.
# train_indices = np.random.choice(np.arange(N_regular, N_regular + N_random), size = train_size, replace = False)
# test_indices = np.setdiff1d(np.arange(N_regular, N_regular + N_random), train_indices)
# test_indices = np.random.choice(test_indices, size = test_size, replace = False)



PATH_CURRENT = "../"

import sys
sys.path.append(PATH_CURRENT + "SRC")

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import time
import tracemalloc

import data_construction
import ph
import model
import ml
import ph_ml
import nn_shallow
import nn_deep
import point_net
import plots





# Hyperparameters.

# Data.
N_regular = 480 # There is 8 shapes, 4 convex (triangle, square, pentagon, circle) and 4 concave (concave triangle, square, pentagon, and annulus).
N_random = 480 # First 50% shapes are convex, the remaining 50% are concave.
n = 5000 # Number of point cloud points.

# General.
train_size = 400 # 400
test_size = 80 # 80

# PH parameters.
# cubical complex resolution = 20 x 20

# DL parameters.
NUM_EPOCHS = 25 # 25
BATCH_SIZE = 32

print("\n\nChoice of hyperparameters: ")
print("Number of 'regular' point clouds N_regular = ", N_regular)
print("Number of 'random' point clouds N_regular = ", N_random)
print("Numper of point cloud points n = ", n)
print("NUM_EPOCHS = ", NUM_EPOCHS)
print("BATCH_SIZE = ", BATCH_SIZE)
# alphashape alpha=0.95
# num_x_pixels = 20, num_y_pixels = 20

PATH_RESULTS = PATH_CURRENT + "results/convexity/"
if not os.path.isdir(PATH_RESULTS):
    os.mkdir(PATH_RESULTS)




# Data.

# print("\n\nConstructing and storing the data...")
# data_pc_regular, labels_regular = data_construction.build_dataset_convexity_regular(N_regular, n)
# data_pc_random, labels_random = data_construction.build_dataset_convexity_random(N_random, n)
# data_pc = np.concatenate((data_pc_regular, data_pc_random))
# labels = np.concatenate((labels_regular, labels_random))   
# with open(PATH_CURRENT  + "DATASETS/convexity/point_clouds.pkl", "wb") as f:
#     pickle.dump(data_pc, f)     
# with open(PATH_CURRENT + "DATASETS/convexity/labels.pkl", "wb") as f:
#     pickle.dump(labels, f)

# print("\n\nImporting the data...")
data_pc = data_construction.import_point_clouds(name = "convexity")
labels = data_construction.import_labels(name = "convexity")
labels = labels.astype(int) # Otherwise, the problem is recognized as regression, and mean squared error is calculated.


# Train and test indices.
train_indices = np.random.choice(np.arange(N_regular, N_regular + N_random), size = train_size, replace = False)
test_indices = np.setdiff1d(np.arange(N_regular, N_regular + N_random), train_indices)
test_indices = np.random.choice(test_indices, size = test_size, replace = False)


# PH.
print("\n\nCalculating PH signature on the height filtration from nine directions...")
start_time = time.time()
tracemalloc.start()
data_ph = ph.calculate_ph_height_point_clouds(data_pc)
data_ph_train = data_ph[train_indices] 
data_ph_test = data_ph[test_indices] 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph = np.around(end_time - start_time, 2)
memory_ph = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph, "seconds.")
print("Memory = ", memory_ph, "kB.\n\n")


# Distance matrices.
print("\n\nCalculating distance matrices (input for ML and NN)...")
start_time = time.time()
tracemalloc.start()
data_dis_mat_flat = data_construction.calculate_distance_matrices_flat(data_pc)
data_dis_mat_flat_train = data_dis_mat_flat[train_indices] 
data_dis_mat_flat_test = data_dis_mat_flat[test_indices] 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_dis_mat = np.around(end_time - start_time, 2)
memory_dis_mat = np.around(memory_current/1000, 2)
print("Total runtime = ", time_dis_mat, "seconds.")
print("Memory = ", memory_dis_mat, "kB.\n\n")


# 3D point clouds.
print("\n\nCalculating 3D point clouds (input for PointNet...")
start_time = time.time()
tracemalloc.start()
data_pc_3d = data_construction.calculate_3d_point_clouds(data_pc)
data_pc_3d_train = data_pc_3d[train_indices] 
data_pc_3d_test = data_pc_3d[test_indices]  
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_3d_pc = np.around(end_time - start_time, 2)
memory_3d_pc = np.around(memory_current/1000, 2)
print("Total runtime = ", time_3d_pc, "seconds.")
print("Memory = ", memory_3d_pc, "kB.\n\n")


# Labels.
labels_train = labels[train_indices]
labels_test = labels[test_indices]



# Build models (default, or with hyperparameters tuned on train data).

print("\n\nTuning the hyperparameters of PH...")
start_time = time.time()
tracemalloc.start()
model_ph = ml.tune_hyperparameters(data_ph_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph_tune = np.around(end_time - start_time, 2)
memory_ph_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph_tune, "seconds.")
print("Memory = ", memory_ph_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of ML on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_ml = ml.tune_hyperparameters(data_dis_mat_flat_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_tune = np.around(end_time - start_time, 2)
memory_ml_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_tune, "seconds.")
print("Memory = ", memory_ml_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of NN shallow on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_nn_shallow = nn_shallow.tune_hyperparameters(data_dis_mat_flat_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_nn_shallow_tune = np.around(end_time - start_time, 2)
memory_nn_shallow_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_nn_shallow_tune, "seconds.")
print("Memory = ", memory_nn_shallow_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of NN deep on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_nn_deep = nn_deep.tune_hyperparameters(data_dis_mat_flat_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_nn_deep_tune = np.around(end_time - start_time, 2)
memory_nn_deep_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_nn_deep_tune, "seconds.")
print("Memory = ", memory_nn_deep_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of PointNet on 3D point clouds...")
start_time = time.time()
tracemalloc.start()
model_point_net = point_net.tune_hyperparameters(data_pc_3d_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_point_net_tune = np.around(end_time - start_time, 2)
memory_point_net_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_point_net_tune, "seconds.")
print("Memory = ", memory_point_net_tune, "kB.\n\n")



# Training the models.

print("\n\nTraining PH...")
start_time = time.time()
tracemalloc.start()
model_trained_ph, _ = model.fit(data_ph_train, labels_train, model_ph)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph_train = np.around(end_time - start_time, 2)
memory_ph_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph_train, "seconds.")
print("Memory = ", memory_ph_train, "kB.\n\n")

print("\n\nTraining ML on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_trained_ml, _ = model.fit(data_dis_mat_flat_train, labels_train, model_ml)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_train = np.around(end_time - start_time, 2)
memory_ml_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_train, "seconds.")
print("Memory = ", memory_ml_train, "kB.\n\n")

print("\n\nTraining NN shallow on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_trained_nn_shallow, _ = model.fit(data_dis_mat_flat_train, labels_train, model_nn_shallow, num_train_iter = 25) 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_nn_shallow_train = np.around(end_time - start_time, 2)
memory_nn_shallow_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_nn_shallow_train, "seconds.")
print("Memory = ", memory_nn_shallow_train, "kB.\n\n")

print("\n\nTraining NN deep on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_trained_nn_deep, _ = model.fit(data_dis_mat_flat_train, labels_train, model_nn_deep, num_train_iter = 25) 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_nn_deep_train = np.around(end_time - start_time, 2)
memory_nn_deep_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_nn_deep_train, "seconds.")
print("Memory = ", memory_nn_deep_train, "kB.\n\n")

print("\n\nTraining PointNet on 3D point clouds...")
start_time = time.time()
tracemalloc.start()
model_trained_point_net, _ = model.fit(data_pc_3d_train, labels_train, model_point_net, num_train_iter = 25)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_point_net_train = np.around(end_time - start_time, 2)
memory_point_net_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_point_net_train, "seconds.")
print("Memory = ", memory_point_net_train, "kB.\n\n")



# Visualize usage of computational resources.
pipelines = ["PH", "ML", "NN shallow", "NN deep", "PointNet"]
comp_times_data = [time_ph, time_dis_mat, time_dis_mat, time_dis_mat, time_3d_pc]
comp_times_tune = [time_ph_tune, time_ml_tune, time_nn_shallow_tune, time_nn_deep_tune, time_point_net_tune]
comp_times_train = [time_ph_train, time_ml_train, time_nn_shallow_train, time_nn_deep_train, time_point_net_train]
memory_data = [memory_ph, memory_dis_mat, memory_dis_mat, memory_dis_mat, memory_3d_pc]
memory_tune = [memory_ph_tune, memory_ml_tune, memory_nn_shallow_tune, memory_nn_deep_tune, memory_point_net_tune]
memory_train = [memory_ph_train, memory_ml_train, memory_nn_shallow_train, memory_nn_deep_train, memory_point_net_train]

width = 0.5 # the width of the bars: can also be len(x) sequence

fig, axes = plt.subplots()
axes.bar(pipelines, comp_times_train, width, label = "train")
axes.bar(pipelines, comp_times_tune, width, label = "tune")
axes.bar(pipelines, comp_times_data, width, label = "data")
axes.set_xlabel("pipeline", fontsize = 20)
axes.set_ylabel("computation time (s)", fontsize = 20)
# axes.legend(fontsize = 20, bbox_to_anchor = (1.1, 1)) 
plt.savefig(PATH_RESULTS + "computation_times_random_random", bbox_inches = "tight")
plt.show()

fig, axes = plt.subplots()
axes.bar(pipelines, memory_train, width, label = "train")
axes.bar(pipelines, memory_tune, width, label = "tune")
axes.bar(pipelines, memory_data, width, label = "data")
axes.set_xlabel("pipeline", fontsize = 20)
axes.set_ylabel("memory (kB)", fontsize = 20)
# axes.legend(fontsize = 20, bbox_to_anchor = (1.1, 1)) 
legend_handles, legend_labels = axes.get_legend_handles_labels()
axes.legend(reversed(legend_handles), reversed(legend_labels), fontsize = 20, bbox_to_anchor = (1.1, 1))
plt.savefig(PATH_RESULTS + "memory_random_random", bbox_inches = "tight")
plt.show()



print("\n\nEvaluation of accuracies of all models...")
start_time = time.time()

acc_ph = model.get_score(data_ph_test, labels_test, model_trained_ph)
acc_ml = model.get_score(data_dis_mat_flat_test, labels_test, model_trained_ml)
acc_nn_shallow = model.get_score(data_dis_mat_flat_test, labels_test, model_trained_nn_shallow)
acc_nn_deep = model.get_score(data_dis_mat_flat_test, labels_test, model_trained_nn_deep)
acc_point_net = model.get_score(data_pc_3d_test, labels_test, model_trained_point_net)
print("acc_ph = ", acc_ph)
print("acc_ml = ", acc_ml)
print("acc_nn_shallow = ", acc_nn_shallow)
print("acc_nn_deep = ", acc_nn_deep)
print("acc_point_net = ", acc_point_net)

# We need to store all the accuracies, in order to be able to plot accuracies accross different paradigms in a single figure.
with open(PATH_RESULTS + "acc_ph_random_random.pkl", "wb") as f:
    pickle.dump(acc_ph, f) 
with open(PATH_RESULTS + "acc_ml_random_random.pkl", "wb") as f:
    pickle.dump(acc_ml, f)
with open(PATH_RESULTS + "acc_nn_shallow_random_random.pkl", "wb") as f:
    pickle.dump(acc_nn_shallow, f) 
with open(PATH_RESULTS + "acc_nn_deep_random_random.pkl", "wb") as f:
    pickle.dump(acc_nn_deep, f) 
with open(PATH_RESULTS + "acc_point_net_random_random.pkl", "wb") as f:
    pickle.dump(acc_point_net, f) 

# Quick visualization of the accuracies for this paradigm (train = random, test = random).
pipelines = ["PH", "ML", "NN shallow", "NN deep", "PointNet"]
accs = {}
accs["PH"] = acc_ph
accs["ML"] = acc_ml
accs["NN shallow"] = acc_nn_shallow
accs["NN deep"] = acc_nn_deep
accs["PointNet"] = acc_point_net
fig, axes = plots.plot_bar_chart(["pipeline"], accs, pipelines)
fig.set_size_inches(3, 3)
legend = axes.legend(pipelines, ncol = 1, fontsize = 10, bbox_to_anchor=(1.10, 1)) 
plt.savefig(PATH_RESULTS + "accs_random_random", bbox_inches = "tight")



# Confusion matrices.
print("\n\nPlotting confusion matrices...")
model.plot_confusion_matrix(data_ph_test, labels_test, None, model_trained_ph, output_path = PATH_RESULTS + "confusion_matrix_ph_random_random")
model.plot_confusion_matrix(data_dis_mat_flat_test, labels_test, None, model_trained_ml, output_path = PATH_RESULTS + "confusion_matrix_ml_random_random")
model.plot_confusion_matrix(data_dis_mat_flat_test, labels_test, None, model_trained_nn_shallow, output_path = PATH_RESULTS + "confusion_matrix_nn_shallow_random_random")
model.plot_confusion_matrix(data_dis_mat_flat_test, labels_test, None, model_trained_nn_deep, output_path = PATH_RESULTS + "confusion_matrix_nn_deep_random_random")
model.plot_confusion_matrix(data_pc_3d_test, labels_test, None, model_trained_point_net, output_path = PATH_RESULTS + "confusion_matrix_point_net_random_random")



# Wrong predictions.
print("\n\nVisualizing wrong predictions...")
data_pc_test = data_pc[test_indices]
model.visualize_incorrect_predictions(data_pc_test, data_ph_test, labels_test, None, model_trained_ph, output_path = PATH_RESULTS + "wrong_predictions_ph_random_random")
model.visualize_incorrect_predictions(data_pc_test, data_dis_mat_flat_test, labels_test, None, model_trained_ml, output_path = PATH_RESULTS + "wrong_predictions_ml_random_random")
model.visualize_incorrect_predictions(data_pc_test, data_dis_mat_flat_test, labels_test, None, model_trained_nn_shallow, output_path = PATH_RESULTS + "wrong_predictions_nn_shallow_random_random")
model.visualize_incorrect_predictions(data_pc_test, data_dis_mat_flat_test, labels_test, None, model_trained_nn_deep, output_path = PATH_RESULTS + "wrong_predictions_nn_deep_random_random")
model.visualize_incorrect_predictions(data_pc_test, data_pc_3d_test, labels_test, None, model_trained_point_net, output_path = PATH_RESULTS + "wrong_predictions_point_net_random_random")
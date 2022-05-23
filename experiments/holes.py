print("\n\nNUMBER OF HOLES: Point clouds ordinal classification via persistent homology (PH) and/or deep learning (DL).\n\n")



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
N = 1000 # 1000
n = 1000 # 1000

# PH parameters.
FIL_COMPLEX = "alpha"
FIL_FUN = "dtm"
DTM_M = 0.03
DTM_P = 1

# DL parameters.
NUM_EPOCHS = 25 # 25
BATCH_SIZE = 32
TRAIN_SIZES = np.linspace(0.1, 1, 10) # np.linspace(0.1, 1, 10)

print("\n\nChoice of hyperparameters: ")
print("Number of point clouds N = ", N)
print("Numper of point cloud points n = ", n)
print("FIL_COMPLEX = ", FIL_COMPLEX)
print("FIL_FUN = ", FIL_FUN)
print("DTM_M = ", DTM_M)
print("DTM_P = ", DTM_P)
print("NUM_EPOCHS = ", NUM_EPOCHS)
print("BATCH_SIZE = ", BATCH_SIZE)
print("TRAIN_SIZES = ", TRAIN_SIZES)

# Other parameters.
# distance matrices number of random point cloud points = 100
# num_cycles = 10 for the simple PH signature
# PI bandwidth in {0.1, 0.5, 1, 10}
# PI weight function ω(x, y) ∈ {1, y, y^2}
# PI resolution = 10 x 10
# PL num_landscapes in {1, 10, max_num_cycles}
# PL resolution = 100
# NN layer widths
# NN depth
# num_splits = 3 in learning and training curves
# cv = 3 number of splits in grid search
# train_size = int(0.8 * num_point_clouds)
# test_size = int(0.2 * num_point_clouds)

PATH_RESULTS = PATH_CURRENT + "/results/holes/"
if not os.path.isdir(PATH_RESULTS):
    os.mkdir(PATH_RESULTS)



# Data.

print("\n\nConstructing the data...")
data_pc, labels, labels_shape = data_construction.build_dataset_holes(N, n)
with open(PATH_CURRENT  + "DATASETS/holes/point_clouds.pkl", "wb") as f:
    pickle.dump(data_pc, f)     
with open(PATH_CURRENT + "DATASETS/holes/labels.pkl", "wb") as f:
    pickle.dump(labels, f) 
with open(PATH_CURRENT + "DATASETS/holes/labels_shape.pkl", "wb") as f:
    pickle.dump(labels_shape, f) 

# print("\n\nImporting the data...")
# data_pc = data_construction.import_point_clouds(name = "holes")
# labels = data_construction.import_labels(name = "holes")



# Train and test data.
num_point_clouds = len(data_pc)
train_size = int(0.8 * num_point_clouds)
test_size = int(0.2 * num_point_clouds)
train_indices = np.random.choice(np.arange(num_point_clouds), size = train_size, replace = False)
non_train_indices = np.setdiff1d(np.arange(num_point_clouds), train_indices)
test_indices = np.random.choice(non_train_indices, size = test_size, replace = False)
with open(PATH_CURRENT + "DATASETS/holes/train_indices.pkl", "wb") as f:
    pickle.dump(train_indices, f)     
with open(PATH_CURRENT + "DATASETS/holes/test_indices.pkl", "wb") as f:
    pickle.dump(test_indices, f)

    

# PDs.
print("\n\nCalculating PDs (input for PH)...")
start_time = time.time()
tracemalloc.start()
_, data_pd = ph.calculate_pds_point_clouds(data_pc, fil_complex = FIL_COMPLEX, fil_fun = FIL_FUN, m = DTM_M, p = DTM_P)
with open(PATH_CURRENT + "DATASETS/holes/persistence_diagrams.pkl", "wb") as f:
    pickle.dump(data_pd, f)      
data_pd_train = data_pd[train_indices] 
data_pd_test = data_pd[test_indices]
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_pd = np.around(end_time - start_time, 2)
memory_pd = np.around(memory_current/1000, 2)
print("Total runtime = ", time_pd, "seconds.")
print("Memory = ", memory_pd, "kB.\n\n")



# PH simple.
print("\n\nCalculating sorted lifespans from PDs (input for simple PH)...")
start_time = time.time()
tracemalloc.start()
data_ph = ph.sorted_lifespans_pds(data_pd, size = 10)
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



# 3D points clouds.
print("\n\nCalculating 3D point clouds (input for PointNet)...")
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
labels_con, label_encoder = data_construction.encode_labels(labels)
labels_con_train = labels_con[train_indices]
labels_con_test = labels_con[test_indices]



# Build models (default, or with hyperparameters tuned on train data).

print("\n\nTuning the hyperparameters of ML on simple PH...")
start_time = time.time()
tracemalloc.start()
model_ml_on_ph = ml.tune_hyperparameters(data_ph_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph_tune = np.around(end_time - start_time, 2)
memory_ml_on_ph_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph_tune, "seconds.")
print("Memory = ", memory_ml_on_ph_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of PH...")
start_time = time.time()
tracemalloc.start()
model_ph_ml = ph_ml.tune_hyperparameters(data_pd_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph_ml_tune = np.around(end_time - start_time, 2)
memory_ph_ml_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph_ml_tune, "seconds.")
print("Memory = ", memory_ph_ml_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of ML on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_ml_on_dis_mat = ml.tune_hyperparameters(data_dis_mat_flat_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_dis_mat_tune = np.around(end_time - start_time, 2)
memory_ml_on_dis_mat_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_dis_mat_tune, "seconds.")
print("Memory = ", memory_ml_on_ph_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of NN shallow on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_nn_shallow = nn_shallow.tune_hyperparameters(data_dis_mat_flat_train, labels_con_train)
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
model_nn_deep = nn_deep.tune_hyperparameters(data_dis_mat_flat_train, labels_con_train)
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
model_point_net = point_net.tune_hyperparameters(data_pc_3d_train, labels_con_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_point_net_tune = np.around(end_time - start_time, 2)
memory_point_net_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_point_net_tune, "seconds.")
print("Memory = ", memory_point_net_tune, "kB.\n\n")



# Learning curves.
print("Plotting the learning curves...")
model.fit_and_plot_learning_curve(data_train = data_ph_train, 
                                  labels_train = labels_train,
                                  model = model_ml_on_ph,
                                  train_sizes = TRAIN_SIZES, 
                                  num_splits = 3, 
                                  output_path = PATH_RESULTS + "learning_curve_ml_on_ph")   
model.fit_and_plot_learning_curve(data_train = data_pd_train, 
                                  labels_train = labels_train,
                                  model = model_ph_ml,
                                  train_sizes = TRAIN_SIZES, 
                                  num_splits = 3, 
                                  output_path = PATH_RESULTS + "learning_curve_ph_ml")   
model.fit_and_plot_learning_curve(data_train = data_dis_mat_flat_train, 
                                  labels_train = labels_train,
                                  model = model_ml_on_dis_mat,
                                  train_sizes = TRAIN_SIZES, 
                                  num_splits = 3, 
                                  output_path = PATH_RESULTS + "learning_curve_ml_on_dis_mat")
model.fit_and_plot_learning_curve(data_train = data_dis_mat_flat_train, 
                                  labels_train = labels_con_train,
                                  model = model_nn_shallow,
                                  train_sizes = TRAIN_SIZES, 
                                  num_splits = 3, 
                                  output_path = PATH_RESULTS + "learning_curve_nn_shallow")   
model.fit_and_plot_learning_curve(data_train = data_dis_mat_flat_train, 
                                  labels_train = labels_con_train,
                                  model = model_nn_deep,
                                  train_sizes = TRAIN_SIZES, 
                                  num_splits = 3, 
                                  output_path = PATH_RESULTS + "learning_curve_nn_deep")   
model.fit_and_plot_learning_curve(data_train = data_pc_3d_train, 
                                  labels_train = labels_con_train,
                                  model = model_point_net,
                                  train_sizes = TRAIN_SIZES, 
                                  num_splits = 3, 
                                  output_path = PATH_RESULTS + "learning_curve_point_net") 
print("\n\n")



# Training curves.
print("Plotting the training curves...")
model.fit_and_plot_training_curve(data_train = data_dis_mat_flat_train, 
                                  labels_train = labels_con_train,
                                  model = model_nn_shallow,
                                  num_epochs = NUM_EPOCHS,
                                  batch_size = BATCH_SIZE,
                                  output_path = PATH_RESULTS + "training_curve_nn_shallow") 
model.fit_and_plot_training_curve(data_train = data_dis_mat_flat_train, 
                                  labels_train = labels_con_train,
                                  model = model_nn_deep,
                                  num_epochs = NUM_EPOCHS,
                                  batch_size = BATCH_SIZE,
                                  output_path = PATH_RESULTS + "training_curve_nn_deep") 
model.fit_and_plot_training_curve(data_train = data_pc_3d_train, 
                                  labels_train = labels_con_train,
                                  model = model_point_net,
                                  num_epochs = NUM_EPOCHS,
                                  batch_size = BATCH_SIZE,
                                  output_path = PATH_RESULTS + "training_curve_point_net") 
print("\n\n")



# Train models.

print("\n\nTraining ML on simple PH...")
start_time = time.time()
tracemalloc.start()
model_trained_ml_on_ph, _ = model.fit(data_ph_train, labels_train, model_ml_on_ph)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph_train = np.around(end_time - start_time, 2)
memory_ml_on_ph_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph_train, "seconds.")
print("Memory = ", memory_ml_on_ph_train, "kB.\n\n")

print("\n\nTraining PH...")
start_time = time.time()
tracemalloc.start()
model_trained_ph_ml, _ = model.fit(data_pd_train, labels_train, model_ph_ml)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph_ml_train = np.around(end_time - start_time, 2)
memory_ph_ml_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph_ml_train, "seconds.")
print("Memory = ", memory_ph_ml_train, "kB.\n\n")

print("\n\nTraining ML on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_trained_ml_on_dis_mat, _ = model.fit(data_dis_mat_flat_train, labels_train, model_ml_on_dis_mat)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_dis_mat_train = np.around(end_time - start_time, 2)
memory_ml_on_dis_mat_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_dis_mat_train, "seconds.")
print("Memory = ", memory_ml_on_dis_mat_train, "kB.\n\n")

print("\n\nTraining NN shallow...")
start_time = time.time()
tracemalloc.start()
model_trained_nn_shallow, _ = model.fit(data_dis_mat_flat_train, labels_con_train, model_nn_shallow, NUM_EPOCHS) 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_nn_shallow_train = np.around(end_time - start_time, 2)
memory_nn_shallow_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_nn_shallow_train, "seconds.")
print("Memory = ", memory_nn_shallow_train, "kB.\n\n")

print("\n\nTraining NN deep...")
start_time = time.time()
tracemalloc.start()
model_trained_nn_deep, _ = model.fit(data_dis_mat_flat_train, labels_con_train, model_nn_deep, NUM_EPOCHS) 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_nn_deep_train = np.around(end_time - start_time, 2)
memory_nn_deep_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_nn_deep_train, "seconds.")
print("Memory = ", memory_nn_deep_train, "kB.\n\n")

print("\n\nTraining PointNet...")
start_time = time.time()
tracemalloc.start()
model_trained_point_net, history_point_net = model.fit(data_pc_3d_train, labels_con_train, model_point_net, NUM_EPOCHS)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_point_net_train = np.around(end_time - start_time, 2)
memory_point_net_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_point_net_train, "seconds.")
print("Memory = ", memory_point_net_train, "kB.\n\n")
model.plot_training_curve_keras(history_point_net, num_epochs = NUM_EPOCHS, output_path = PATH_RESULTS + "training_curve_point_net_history")



# Visualize usage of computational resources.
pipelines = ["PH simple", "PH", "ML", "NN shallow", "NN deep", "PointNet"]
comp_times_data = [time_pd + time_ph, time_pd, time_dis_mat, time_dis_mat, time_dis_mat, time_3d_pc]
comp_times_tune = [time_ml_on_ph_tune, time_ph_ml_tune, time_ml_on_dis_mat_tune, time_nn_shallow_tune, time_nn_deep_tune, time_point_net_tune]
comp_times_train = [time_ml_on_ph_train, time_ph_ml_train, time_ml_on_dis_mat_train, time_nn_shallow_train, time_nn_deep_train, time_point_net_train]
memory_data = [memory_pd + memory_ph, memory_pd, memory_dis_mat, memory_dis_mat, memory_dis_mat, memory_3d_pc]
memory_tune = [memory_ml_on_ph_tune, memory_ph_ml_tune, memory_ml_on_dis_mat_tune, memory_nn_shallow_tune, memory_nn_deep_tune, memory_point_net_tune]
memory_train = [memory_ml_on_ph_train, memory_ph_ml_train, memory_ml_on_dis_mat_train, memory_nn_shallow_train, memory_nn_deep_train, memory_point_net_train]

width = 0.5 # the width of the bars: can also be len(x) sequence

fig, axes = plt.subplots()
axes.bar(pipelines, comp_times_train, width, label = "train")
axes.bar(pipelines, comp_times_tune, width, label = "tune")
axes.bar(pipelines, comp_times_data, width, label = "data")
axes.set_xlabel("pipeline", fontsize = 20)
axes.set_ylabel("computation time (s)", fontsize = 20)
# axes.legend(fontsize = 20, bbox_to_anchor = (1.1, 1)) 
plt.savefig(PATH_RESULTS + "computation_times", bbox_inches = "tight")
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
plt.savefig(PATH_RESULTS + "memory", bbox_inches = "tight")
plt.show()



# Evaluate accuracies.
print("\n\nEvaluation of accuracies of all models...")
start_time = time.time()

acc_ml_on_ph = model.get_score(data_ph_test, labels_test, model_trained_ml_on_ph)
acc_ph_ml = model.get_score(data_pd_test, labels_test, model_trained_ph_ml)
acc_ml_on_dis_mat = model.get_score(data_dis_mat_flat_test, labels_test, model_trained_ml_on_dis_mat)
acc_nn_shallow = model.get_score(data_dis_mat_flat_test, labels_con_test, model_trained_nn_shallow)
acc_nn_deep = model.get_score(data_dis_mat_flat_test, labels_con_test, model_trained_nn_deep)
acc_point_net = model.get_score(data_pc_3d_test, labels_con_test, model_trained_point_net)
print("acc_ml_on_ph = ", acc_ml_on_ph)
print("acc_ph_ml = ", acc_ph_ml)
print("acc_ml_on_dis_mat = ", acc_ml_on_dis_mat)
print("acc_nn_shallow = ", acc_nn_shallow)
print("acc_nn_deep = ", acc_nn_deep)
print("acc_point_net = ", acc_point_net)

end_time = time.time()
print("Total runtime = ", np.around(end_time - start_time, 2), "seconds.\n\n")

pipelines = ["PH simple", "PH", "ML", "NN shallow", "NN deep", "PointNet"]
accs = {}
accs["PH simple"] = acc_ml_on_ph
accs["PH"] = acc_ph_ml
accs["ML"] = acc_ml_on_dis_mat
accs["NN shallow"] = acc_nn_shallow
accs["NN deep"] = acc_nn_deep
accs["PointNet"] = acc_point_net
fig, axes = plots.plot_bar_chart(["pipeline"], accs, pipelines)
# If necessary, resize the figure and move the legend.
fig.set_size_inches(3, 3)
legend = axes.legend(pipelines, ncol = 1, fontsize = 10, bbox_to_anchor=(1.75, 1)) 
plt.savefig(PATH_RESULTS + "accs", bbox_inches = "tight")



# Confusion matrices.
model.plot_confusion_matrix(data_ph_test, labels_test, None, model_trained_ml_on_ph, output_path = PATH_RESULTS + "confusion_matrix_ml_on_ph")
model.plot_confusion_matrix(data_pd_test, labels_test, None, model_trained_ph_ml, output_path = PATH_RESULTS + "confusion_matrix_ph_ml")
model.plot_confusion_matrix(data_dis_mat_flat_test, labels_test, None, model_trained_ml_on_dis_mat, output_path = PATH_RESULTS + "confusion_matrix_ml_on_dis_mat")
model.plot_confusion_matrix(data_dis_mat_flat_test, labels_test, label_encoder, model_trained_nn_shallow, output_path = PATH_RESULTS + "confusion_matrix_nn_shallow")
model.plot_confusion_matrix(data_dis_mat_flat_test, labels_test, label_encoder, model_trained_nn_deep, output_path = PATH_RESULTS + "confusion_matrix_nn_deep")
model.plot_confusion_matrix(data_pc_3d_test, labels_test, label_encoder, model_trained_point_net, output_path = PATH_RESULTS + "confusion_matrix_point_net")



# Noise robustness.
print("\n\nEvaluating the noise robustness...")

data_pc_test = [data_pc[i] for i in test_indices] 
trnsfs = ["std", "trns", "rot", "stretch", "shear", "gauss", "out"]
data_pc_test_trns = data_construction.calculate_point_clouds_under_trnsf(data_pc_test, transformation = "translation")
data_pc_test_rot = data_construction.calculate_point_clouds_under_trnsf(data_pc_test, transformation = "rotation")
data_pc_test_stretch = data_construction.calculate_point_clouds_under_trnsf(data_pc_test, transformation = "stretch")
data_pc_test_shear = data_construction.calculate_point_clouds_under_trnsf(data_pc_test, transformation = "shear")
data_pc_test_gauss = data_construction.calculate_point_clouds_under_trnsf(data_pc_test, transformation = "gaussian")
data_pc_test_out = data_construction.calculate_point_clouds_under_trnsf(data_pc_test, transformation = "outliers")
data_pc_test_trnsfs = {}
data_pc_test_trnsfs["std"] = data_pc_test
data_pc_test_trnsfs["trns"] = data_pc_test_trns
data_pc_test_trnsfs["rot"] = data_pc_test_rot
data_pc_test_trnsfs["stretch"] = data_pc_test_stretch
data_pc_test_trnsfs["shear"] = data_pc_test_shear
data_pc_test_trnsfs["gauss"] = data_pc_test_gauss
data_pc_test_trnsfs["out"] = data_pc_test_out
with open(PATH_CURRENT + "DATASETS/holes/point_clouds_test_trnsfs.pkl", "wb") as f:
    pickle.dump(data_pc_test_trnsfs, f)
    
data_pd_test_trnsfs = {}
for trnsf in trnsfs:
    data_pc_test = data_pc_test_trnsfs[trnsf]
    _, data_pd_test = ph.calculate_pds_point_clouds(data_pc_test, fil_complex = FIL_COMPLEX, fil_fun = FIL_FUN, m = DTM_M, p = DTM_P)
    data_pd_test_trnsfs[trnsf] = data_pd_test
    
# # Transform list of 1-dim PDs with different number of cycles into an array of PDs with the same number of cycles.    
# pd_lengths = [data_pd_test_trnsfs[trnsf].shape[1] for trnsf in trnsfs]
# max_pd_length  = max(pd_lengths) 
# for trnsf in trnsfs:
#     pds = data_pd_test_trnsfs[trnsf]
#     pds_ext = ph.extend_pds_to_length(pds, max_pd_length)
#     data_pd_test_trnsfs[trnsf] = pds_ext
#     print("data_pd_test_trnsfs[", trnsf, "].shape = ", data_pd_test_trnsfs[trnsf].shape)
    
# We need to re-train the model on the adjusted PDs.
# data_pd_train = ph.extend_pds_to_length(data_pd_train, max_pd_length)    
# model_ph_ml.fit(data_pd_test_trnsfs["std"], labels_test)
# model_trained_ph_ml.fit(data_pd_test_trnsfs["std"], labels_test)

data_ph_test_trnsfs = {}
for trnsf in trnsfs:
    data_pd_test = data_pd_test_trnsfs[trnsf]
    data_ph_test = ph.sorted_lifespans_pds(data_pd_test, size = 10)
    data_ph_test_trnsfs[trnsf] = data_ph_test
    
data_pc_3d_test_trnsfs = data_construction.calculate_3d_point_clouds_under_trnsfs(point_clouds_trnsfs = data_pc_test_trnsfs, trnsfs = trnsfs)
data_dis_mat_flat_test_trnsfs = data_construction.calculate_distance_matrices_flat_under_trnsfs(point_clouds_trnsfs = data_pc_test_trnsfs, trnsfs = trnsfs)    

accs_trnsfs_ml_on_ph = model.get_scores_under_trnsfs(data_ph_test_trnsfs, trnsfs, labels_test, model_trained_ml_on_ph)
accs_trnsfs_ph_ml = model.get_scores_under_trnsfs(data_pd_test_trnsfs, trnsfs, labels_test, model_trained_ph_ml)
accs_trnsfs_ml_on_dis_mat = model.get_scores_under_trnsfs(data_dis_mat_flat_test_trnsfs, trnsfs, labels_test, model_trained_ml_on_dis_mat)
accs_trnsfs_nn_shallow = model.get_scores_under_trnsfs(data_dis_mat_flat_test_trnsfs, trnsfs, labels_con_test, model_trained_nn_shallow)
accs_trnsfs_nn_deep = model.get_scores_under_trnsfs(data_dis_mat_flat_test_trnsfs, trnsfs, labels_con_test, model_trained_nn_deep)
accs_trnsfs_point_net = model.get_scores_under_trnsfs(data_pc_3d_test_trnsfs, trnsfs, labels_con_test, model_trained_point_net)
transformations = ["original", "translation", "rotation", "stretch", "shear", "gaussian", "outliers"]
pipelines = ["PH simple", "PH", "ML", "NN shallow", "NN deep", "PointNet"]
accs = {}
accs["PH simple"] = accs_trnsfs_ml_on_ph
accs["PH"] = accs_trnsfs_ph_ml
accs["ML"] = accs_trnsfs_ml_on_dis_mat
accs["NN shallow"] = accs_trnsfs_nn_shallow
accs["NN deep"] = accs_trnsfs_nn_deep
accs["PointNet"] = accs_trnsfs_point_net
fig = plots.plot_bar_chart(transformations, accs, pipelines)
plt.savefig(PATH_RESULTS + "accs_trnsfs", bbox_inches = "tight")
print("\n\nCURVATURE: Point clouds regression via persistent homology (PH) and/or deep learning (DL).\n\n")



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
num_train_curvatures = 101 # 101
num_point_clouds_train_curvature = 10 # 100
num_test_point_clouds = 100 # 100
n = 500 # 1000

# PH parameters.

# DL parameters.
NUM_EPOCHS = 25 # 25
BATCH_SIZE = 32

print("\n\nChoice of hyperparameters: ")
# print("Number of point clouds N = ", N)
print("Numper of point cloud points n = ", n)
print("NUM_EPOCHS = ", NUM_EPOCHS)
print("BATCH_SIZE = ", BATCH_SIZE)

# Other parameters.
# num_cycles = 10 for the simple PH signature
# PI bandwidth in {0.1, 0.5, 1, 10}
# PI  weight function ω(x, y) ∈ {1, y, y^2}
# PI resolution
# PL num_landscapes
# PL resolution
# NN layer widths
# NN depth
# num_splits = 3 in learning and training curves
# cv = 3 number of splits in grid search

PATH_RESULTS = PATH_CURRENT + "results/curvature/"
if not os.path.isdir(PATH_RESULTS):
    os.mkdir(PATH_RESULTS)




# Data.

print("\n\nConstructing the data...")
start_time = time.time()
tracemalloc.start()
data_pc, labels, data_dis_mat = data_construction.build_dataset_curvature(n, num_train_curvatures, num_point_clouds_train_curvature, num_test_point_clouds)
with open(PATH_CURRENT  + "DATASETS/curvature/point_clouds.pkl", "wb") as f:
    pickle.dump(data_pc, f)     
with open(PATH_CURRENT + "DATASETS/curvature/labels.pkl", "wb") as f:
    pickle.dump(labels, f) 

# print("\n\nImporting the data...")
# data_pc = data_construction.import_point_clouds(name = "curvature")
# data_dis_mat = data_construction.import_distance_matrices(name = "curvature")
# labels = data_construction.import_labels(name = "curvature")
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print("Total runtime = ", np.around(end_time - start_time, 2), "seconds.")
print("Memory = ", np.around(memory_current/1000, 2), "kB.\n\n")



# Train and test data.
train_size = num_train_curvatures * num_point_clouds_train_curvature
test_size = num_test_point_clouds
train_indices = np.arange(train_size)
test_indices = np.arange(train_size, train_size + test_size)
print("train_indices.shape = ", train_indices.shape)
print("test_indices.shape = ", test_indices.shape)
with open(PATH_CURRENT + "DATASETS/curvature/train_indices.pkl", "wb") as f:
    pickle.dump(train_indices, f)     
with open(PATH_CURRENT + "DATASETS/curvature/test_indices.pkl", "wb") as f:
    pickle.dump(test_indices, f) 


    
# PDs.
print("\n\nCalculating PDs (input for PH)...")
start_time = time.time()
tracemalloc.start()
data_pd0, data_pd1 = ph.calculate_pds_distance_matrices_ripser(data_dis_mat)
with open(PATH_CURRENT + "DATASETS/curvature/persistence_diagrams_dim0.pkl", "wb") as f:
    pickle.dump(data_pd0, f)   
with open(PATH_CURRENT + "DATASETS/curvature/persistence_diagrams_dim1.pkl", "wb") as f:
    pickle.dump(data_pd1, f)  
data_pd0_train = data_pd0[train_indices] 
data_pd0_test = data_pd0[test_indices] 
data_pd1_train = data_pd1[train_indices] 
data_pd1_test = data_pd1[test_indices] 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_pd = np.around(end_time - start_time, 2)
memory_pd = np.around(memory_current/1000, 2)
print("Total runtime = ", time_pd, "seconds.")
print("Memory = ", memory_pd, "kB.\n\n")



# PH.
print("\n\nCalculating sorted lifespans from 0-dim PDs (input for simple PH)...")
start_time = time.time()
tracemalloc.start()
data_ph0 = ph.sorted_lifespans_pds(data_pd0, size = n)
data_ph0_train = data_ph0[train_indices] 
data_ph0_test = data_ph0[test_indices] 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph0 = np.around(end_time - start_time, 2)
memory_ph0 = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph0, "seconds.")
print("Memory = ", memory_ph0, "kB.\n\n")

print("\n\nCalculating sorted lifespans from 1-dim PDs (input for simple PH)...")
start_time = time.time()
tracemalloc.start()
data_ph1 = ph.sorted_lifespans_pds(data_pd1, size = n)
data_ph1_train = data_ph1[train_indices] 
data_ph1_test = data_ph1[test_indices] 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph1 = np.around(end_time - start_time, 2)
memory_ph1 = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph1, "seconds.")
print("Memory = ", memory_ph1, "kB.\n\n")
    
    
# Visualize discriminative cycles.
Ks = np.linspace(-2, 2, num = num_train_curvatures)
K_mid_neg = Ks[int(0.25 * num_train_curvatures)]
K_mid_pos = Ks[int(0.75 * num_train_curvatures)]
Ks = [-2, K_mid_neg, 0, K_mid_pos, 2]
lifespans = np.zeros((len(Ks), n))
num_point_clouds = len(data_pc)
for k, K in enumerate(Ks):
    lifespans_K = np.zeros(n)
    num_point_clouds_K = 0
    for i in range(num_point_clouds):
        if labels[i] == K:
            lifespans_K = lifespans_K + data_ph0[i]
            num_point_clouds_K = num_point_clouds_K + 1
    lifespans_K = lifespans_K / num_point_clouds_K
    lifespans[k] = lifespans_K    
fig, axes = plt.subplots()  
for k in range(len(Ks)):
    axes.plot(lifespans[k], label = "K = %.2f" %Ks[k])
axes.legend()
fig.savefig(PATH_RESULTS + "lifespans_avg_data_ph0", bbox_inches = "tight")
plt.show()    



# data_ph0_longest = data_ph0[:, int(n/2):]
# data_ph1_longest = data_ph1[:, int(n/2):]
data_ph0_longest = data_ph0[:, :10]
data_ph1_longest = data_ph1[:, :10]
data_ph0_longest_train = data_ph0_longest[train_indices] 
data_ph0_longest_test = data_ph0_longest[test_indices] 
data_ph1_longest_train = data_ph1_longest[train_indices] 
data_ph1_longest_test = data_ph1_longest[test_indices] 
    
    
    
# Distance matrices.
print("\n\nCalculating distance matrices (input for ML and NN)...")
start_time = time.time()
tracemalloc.start() 
data_dis_mat_longest = data_dis_mat[:, 0:100, 0:100]
data_dis_mat_flat = data_construction.flatten_symmetric_matrices(data_dis_mat_longest)
with open(PATH_CURRENT + "DATASETS/curvature/distance_matrices_flat.pkl", "wb") as f:
    pickle.dump(data_dis_mat_flat, f)
data_dis_mat_flat_train = data_dis_mat_flat[train_indices] 
data_dis_mat_flat_test = data_dis_mat_flat[test_indices] 
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_dis_mat = np.around(end_time - start_time, 2)
memory_dis_mat = np.around(memory_current/1000, 2)
print("Total runtime = ", time_dis_mat, "seconds.")
print("Memory = ", memory_dis_mat, "kB.\n\n")
# !!! Note that in this case, we only take into account the flattening of the distance matrices that are already calculated together with the point clouds.


# 3D point clouds.
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



# Build models (default, or with hyperparameters tuned on train data).

print("\n\nTuning the hyperparameters of ML on simple 0-dim PH...")
start_time = time.time()
tracemalloc.start()
model_ml_on_ph0 = ml.tune_hyperparameters(data_ph0_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph0_tune = np.around(end_time - start_time, 2)
memory_ml_on_ph0_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph0_tune, "seconds.")
print("Memory = ", memory_ml_on_ph0_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of ML on simple 1-dim PH...")
start_time = time.time()
tracemalloc.start()
model_ml_on_ph1 = ml.tune_hyperparameters(data_ph1_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph1_tune = np.around(end_time - start_time, 2)
memory_ml_on_ph1_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph1_tune, "seconds.")
print("Memory = ", memory_ml_on_ph1_tune, "kB.\n\n")                              

print("\n\nTuning the hyperparameters of ML on simple and 0-dim PH, but only the longest intervals...")
start_time = time.time()
tracemalloc.start()
model_ml_on_ph0_longest = ml.tune_hyperparameters(data_ph0_longest_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph0_longest_tune = np.around(end_time - start_time, 2)
memory_ml_on_ph0_longest_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph0_longest_tune, "seconds.")
print("Memory = ", memory_ml_on_ph0_longest_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of ML ons simple and 1-dim PH, but only the longest intervals...")
start_time = time.time()
tracemalloc.start()
model_ml_on_ph1_longest = ml.tune_hyperparameters(data_ph1_longest_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph1_longest_tune = np.around(end_time - start_time, 2)
memory_ml_on_ph1_longest_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph1_longest_tune, "seconds.")
print("Memory = ", memory_ml_on_ph1_longest_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of 0-dim PH...")
start_time = time.time()
tracemalloc.start()
model_ph0_ml = ph_ml.tune_hyperparameters(data_pd0_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph0_ml_tune = np.around(end_time - start_time, 2)
memory_ph0_ml_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph0_ml_tune, "seconds.")
print("Memory = ", memory_ph0_ml_tune, "kB.\n\n")

print("\n\nTuning the hyperparameters of 1-dim PH...")
start_time = time.time()
tracemalloc.start()
model_ph1_ml = ph_ml.tune_hyperparameters(data_pd1_train, labels_train)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph1_ml_tune = np.around(end_time - start_time, 2)
memory_ph1_ml_tune = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph1_ml_tune, "seconds.")
print("Memory = ", memory_ph1_ml_tune, "kB.\n\n")

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
print("Memory = ", memory_ml_on_dis_mat_tune, "kB.\n\n")

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



# Train models.

print("\n\nTraining ML on simple 0-dim PH...")
start_time = time.time()
tracemalloc.start()
model_trained_ml_on_ph0, _ = model.fit(data_ph0_train, labels_train, model_ml_on_ph0)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph0_train = np.around(end_time - start_time, 2)
memory_ml_on_ph0_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph0_train, "seconds.")
print("Memory = ", memory_ml_on_ph0_train, "kB.\n\n")

print("\n\nTraining ML on simple 1-dim PH...")
start_time = time.time()
tracemalloc.start()
model_trained_ml_on_ph1, _ = model.fit(data_ph1_train, labels_train, model_ml_on_ph1)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph1_train = np.around(end_time - start_time, 2)
memory_ml_on_ph1_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph1_train, "seconds.")
print("Memory = ", memory_ml_on_ph1_train, "kB.\n\n")

print("\n\nTraining ML on simple 0-dim PH, but only the longest intervals...")
start_time = time.time()
tracemalloc.start()
model_trained_ml_on_ph0_longest, _ = model.fit(data_ph0_longest_train, labels_train, model_ml_on_ph0_longest)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph0_longest_train = np.around(end_time - start_time, 2)
memory_ml_on_ph0_longest_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph0_longest_train, "seconds.")
print("Memory = ", memory_ml_on_ph0_longest_train, "kB.\n\n")

print("\n\nTraining ML on simple 1-dim PH, but only the longest intervals...")
start_time = time.time()
tracemalloc.start()
model_trained_ml_on_ph1_longest, _ = model.fit(data_ph1_longest_train, labels_train, model_ml_on_ph1_longest)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ml_on_ph1_longest_train = np.around(end_time - start_time, 2)
memory_ml_on_ph1_longest_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ml_on_ph1_longest_train, "seconds.")
print("Memory = ", memory_ml_on_ph1_longest_train, "kB.\n\n")

print("\n\nTrainng 0-dim PH...")
start_time = time.time()
tracemalloc.start()
model_trained_ph0_ml, _ = model.fit(data_pd0_train, labels_train, model_ph0_ml)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph0_ml_train = np.around(end_time - start_time, 2)
memory_ph0_ml_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph0_ml_train, "seconds.")
print("Memory = ", memory_ph0_ml_train, "kB.\n\n")

print("\n\nTraining 1-dim PH...")
start_time = time.time()
tracemalloc.start()
model_trained_ph1_ml, _ = model.fit(data_pd1_train, labels_train, model_ph1_ml)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_ph1_ml_train = np.around(end_time - start_time, 2)
memory_ph1_ml_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_ph1_ml_train, "seconds.")
print("Memory = ", memory_ph1_ml_train, "kB.\n\n")

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

print("\n\nTraining NN shallow on distance matrices...")
start_time = time.time()
tracemalloc.start()
model_trained_nn_shallow, _ = model.fit(data_dis_mat_flat_train, labels_train, model_nn_shallow, num_train_iter = NUM_EPOCHS) 
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
model_trained_nn_deep, _ = model.fit(data_dis_mat_flat_train, labels_train, model_nn_deep, num_train_iter = NUM_EPOCHS) 
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
model_trained_point_net, _ = model.fit(data_pc_3d_train, labels_train, model_point_net, num_train_iter = NUM_EPOCHS)
end_time = time.time()
memory_current, memory_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_point_net_train = np.around(end_time - start_time, 2)
memory_point_net_train = np.around(memory_current/1000, 2)
print("Total runtime = ", time_point_net_train, "seconds.")
print("Memory = ", memory_point_net_train, "kB.\n\n")



# Visualize usage of computational resources.
pipelines = ["PH0 simple", "PH1 simple", "PH0 simple 10", "PH1 simple 10", "PH0", "PH1", "ML", "NN shallow", "NN deep", "PointNet"]
comp_times_data = [time_pd + time_ph0, time_pd + time_ph1, time_pd + time_ph0, time_pd + time_ph1, time_pd, time_pd, time_dis_mat, time_dis_mat, time_dis_mat, time_3d_pc]
comp_times_tune = [time_ml_on_ph0_tune, time_ml_on_ph1_tune, time_ml_on_ph0_longest_tune, time_ml_on_ph1_longest_tune, time_ph0_ml_tune, time_ph1_ml_tune, time_ml_on_dis_mat_tune, time_nn_shallow_tune, time_nn_deep_tune, time_point_net_tune]
comp_times_train = [time_ml_on_ph0_train, time_ml_on_ph1_train, time_ml_on_ph0_longest_train, time_ml_on_ph1_longest_train, time_ph0_ml_train, time_ph1_ml_train, time_ml_on_dis_mat_train, time_nn_shallow_train, time_nn_deep_train, time_point_net_train]
memory_data = [memory_pd + memory_ph0, memory_pd + memory_ph1, memory_pd + memory_ph0, memory_pd + memory_ph1, memory_pd, memory_pd, memory_dis_mat, memory_dis_mat, memory_dis_mat, memory_3d_pc]
memory_tune = [memory_ml_on_ph0_tune, memory_ml_on_ph1_tune, memory_ml_on_ph0_longest_tune, memory_ml_on_ph1_longest_tune, memory_ph0_ml_tune, memory_ph1_ml_tune, memory_ml_on_dis_mat_tune, memory_nn_shallow_tune, memory_nn_deep_tune, memory_point_net_tune]
memory_train = [memory_ml_on_ph0_train, memory_ml_on_ph1_train, memory_ml_on_ph0_longest_train, memory_ml_on_ph1_longest_train, memory_ph0_ml_train, memory_ph1_ml_train, memory_ml_on_dis_mat_train, memory_nn_shallow_train, memory_nn_deep_train, memory_point_net_train]

width = 0.5 # the width of the bars: can also be len(x) sequence

fig, axes = plt.subplots()
axes.bar(pipelines, comp_times_train, width, label = "train")
axes.bar(pipelines, comp_times_tune, width, label = "tune")
axes.bar(pipelines, comp_times_data, width, label = "data")
axes.set_xticklabels(pipelines, rotation = 45, ha = "right")
axes.set_xlabel("pipeline", fontsize = 20)
axes.set_ylabel("computation time (s)", fontsize = 20)
# axes.legend(fontsize = 20, bbox_to_anchor = (1.1, 1)) 
plt.savefig(PATH_RESULTS + "computation_times", bbox_inches = "tight")
plt.show()

fig, axes = plt.subplots()
axes.bar(pipelines, memory_train, width, label = "train")
axes.bar(pipelines, memory_tune, width, label = "tune")
axes.bar(pipelines, memory_data, width, label = "data")
axes.set_xticklabels(pipelines, rotation = 45, ha = "right")
axes.set_xlabel("pipeline", fontsize = 20)
axes.set_ylabel("memory (kB)", fontsize = 20)
# axes.legend(fontsize = 20, bbox_to_anchor = (1.1, 1)) 
legend_handles, legend_labels = axes.get_legend_handles_labels()
axes.legend(reversed(legend_handles), reversed(legend_labels), fontsize = 20, bbox_to_anchor = (1.1, 1))
plt.savefig(PATH_RESULTS + "memory", bbox_inches = "tight")
plt.show()



# Mean squared errors.
print("\n\nEvaluation of all models...")
start_time = time.time()
mse_ml_on_ph0 = model.get_score(data_ph0_test, labels_test, model_trained_ml_on_ph0) 
mse_ml_on_ph1 = model.get_score(data_ph1_test, labels_test, model_trained_ml_on_ph1)
mse_ml_on_ph0_longest = model.get_score(data_ph0_longest_test, labels_test, model_trained_ml_on_ph0_longest) 
mse_ml_on_ph1_longest = model.get_score(data_ph1_longest_test, labels_test, model_trained_ml_on_ph1_longest)
mse_ph0_ml = model.get_score(data_pd0_test, labels_test, model_trained_ph0_ml) 
mse_ph1_ml = model.get_score(data_pd1_test, labels_test, model_trained_ph1_ml)
mse_ml = model.get_score(data_dis_mat_flat_test, labels_test, model_trained_ml_on_dis_mat)
mse_nn_shallow = model.get_score(data_dis_mat_flat_test, labels_test, model_trained_nn_shallow)
mse_nn_deep = model.get_score(data_dis_mat_flat_test, labels_test, model_trained_nn_deep)
mse_point_net = model.get_score(data_pc_3d_test, labels_test, model_trained_point_net)
print("mse_ml_on_ph0 = ", mse_ml_on_ph0)
print("mse_ml_on_ph1 = ", mse_ml_on_ph1)
print("mse_ml_on_ph0_longest = ", mse_ml_on_ph0_longest)
print("mse_ml_on_ph1_longest = ", mse_ml_on_ph1_longest)
print("mse_ph0_ml = ", mse_ph0_ml)
print("mse_ph1_ml = ", mse_ph1_ml)
print("mse_ml_on_dis_mat = ", mse_ml)
print("mse_nn_shallow = ", mse_nn_shallow)
print("mse_nn_deep = ", mse_nn_deep)
print("mse_point_net = ", mse_point_net)
end_time = time.time()
print("Total runtime = ", np.around(end_time - start_time, 2), "seconds.\n\n")



# Regression lines.
print("\n\nPlotting regression lines of all models...")
fig = model.plot_regression_line(data_ph0_test, labels_test, model_trained_ml_on_ph0)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_ph0", bbox_inches = "tight")
fig = model.plot_regression_line(data_ph1_test, labels_test, model_trained_ml_on_ph1)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_ph1", bbox_inches = "tight")
fig = model.plot_regression_line(data_ph0_longest_test, labels_test, model_trained_ml_on_ph0_longest)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_ph0_longest", bbox_inches = "tight")
fig = model.plot_regression_line(data_ph1_longest_test, labels_test, model_trained_ml_on_ph1_longest)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_ph1_longest", bbox_inches = "tight")
fig = model.plot_regression_line(data_pd0_test, labels_test, model_trained_ph0_ml)
plt.savefig(PATH_RESULTS + "regression_line_ph0_ml", bbox_inches = "tight")
model.plot_regression_line(data_pd1_test, labels_test, model_trained_ph1_ml)
plt.savefig(PATH_RESULTS + "regression_line_ph1_ml", bbox_inches = "tight")
model.plot_regression_line(data_dis_mat_flat_test, labels_test, model_trained_ml_on_dis_mat)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_dis_mat", bbox_inches = "tight")
model.plot_regression_line(data_dis_mat_flat_test, labels_test, model_trained_nn_shallow)
plt.savefig(PATH_RESULTS + "regression_line_nn_shallow", bbox_inches = "tight")
model.plot_regression_line(data_dis_mat_flat_test, labels_test, model_trained_nn_deep)
plt.savefig(PATH_RESULTS + "regression_line_nn_deep", bbox_inches = "tight")
model.plot_regression_line(data_pc_3d_test, labels_test, model_trained_point_net)
plt.savefig(PATH_RESULTS + "regression_line_point_net", bbox_inches = "tight")

fig = model.plot_regression_line_unlimited(data_ph0_test, labels_test, model_trained_ml_on_ph0)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_ph0_unlimited", bbox_inches = "tight")
fig = model.plot_regression_line_unlimited(data_ph1_test, labels_test, model_trained_ml_on_ph1)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_ph1_unlimited", bbox_inches = "tight")
fig = model.plot_regression_line_unlimited(data_ph0_longest_test, labels_test, model_trained_ml_on_ph0_longest)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_ph0_longest_unlimited", bbox_inches = "tight")
fig = model.plot_regression_line_unlimited(data_ph1_longest_test, labels_test, model_trained_ml_on_ph1_longest)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_ph1_longest_unlimited", bbox_inches = "tight")
fig = model.plot_regression_line_unlimited(data_pd0_test, labels_test, model_trained_ph0_ml)
plt.savefig(PATH_RESULTS + "regression_line_ph0_ml_unlimited", bbox_inches = "tight")
model.plot_regression_line_unlimited(data_pd1_test, labels_test, model_trained_ph1_ml)
plt.savefig(PATH_RESULTS + "regression_line_ph1_ml_unlimited", bbox_inches = "tight")
model.plot_regression_line_unlimited(data_dis_mat_flat_test, labels_test, model_trained_ml_on_dis_mat)
plt.savefig(PATH_RESULTS + "regression_line_ml_on_dis_mat_unlimited", bbox_inches = "tight")
model.plot_regression_line_unlimited(data_dis_mat_flat_test, labels_test, model_trained_nn_shallow)
plt.savefig(PATH_RESULTS + "regression_line_nn_shallow_unlimited", bbox_inches = "tight")
model.plot_regression_line_unlimited(data_dis_mat_flat_test, labels_test, model_trained_nn_deep)
plt.savefig(PATH_RESULTS + "regression_line_nn_deep_unlimited", bbox_inches = "tight")
model.plot_regression_line_unlimited(data_pc_3d_test, labels_test, model_trained_point_net)
plt.savefig(PATH_RESULTS + "regression_line_point_net_unlimited", bbox_inches = "tight")
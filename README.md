This folder contains the data and the code for the article "On the Effectiveness of Persistent Homology":

1) The folder DATASETS contains the three sets of point clouds used in the paper for the detection of holes, curvature and convexity.

2) The folder SRC contains the source code for the different pipelines: ph_ml.py, nn_shallow.py, nn_deep.py and point_net.py implement the PH, NN shallow, NN deep and PointNet models used in the paper. Some additional .py scripts help to construct the given and other point cloud data (data_construction.py), calculate persistent homology (ph.py), or are used for plotting (plots.py).

3) The folder experiments contains the scripts that allow to directly replicate the experiments from the paper. The environment is defined in the requirements.txt ("pip install alphashape descartes gudhi ripser tensorflow"). To then replicate the experiments related to the number of holes, for instance, it is sufficient to run "python holes.py".
